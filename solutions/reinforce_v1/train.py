"""Train a REINFORCE rover policy with a learned value baseline."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from model import Reinforce


def export_policy(model: Reinforce, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = copy.deepcopy(model).cpu().eval()
    try:
        from arena.protocol import export_onnx
    except ModuleNotFoundError as exc:
        if exc.name != "arena":
            raise
        # The local environment lacks arena-base. The server uses its validator.
        import onnx
        from onnx import TensorProto, helper as onnx_helper

        n = 2
        example = (
            torch.zeros(n, module.obs_dim), torch.zeros(n, dtype=torch.int64),
            torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.ones(n),
            torch.zeros(n, module.memory_size),
        )
        inputs = (
            "observation", "previous_action", "previous_reward",
            "previous_done", "trial_progress", "trial_start", "memory",
        )
        torch.onnx.export(
            module, example, str(path), opset_version=17, dynamo=False,
            input_names=list(inputs), output_names=["logits", "next_memory"],
            dynamic_axes={name: {0: "batch"} for name in (*inputs, "logits", "next_memory")},
        )
        graph = onnx.load(str(path))

        def ensure_contract_inputs(graph) -> None:
            expected = (
                "observation", "previous_action", "previous_reward", "previous_done",
                "trial_progress", "trial_start", "memory",
            )
            existing = {item.name for item in graph.graph.input}
            if existing == set(expected):
                return
            dtypes = {"previous_action": TensorProto.INT64}
            by_name = {item.name: item for item in graph.graph.input}
            for name in expected:
                if name not in existing:
                    value_info = onnx_helper.make_tensor_value_info(
                        name, dtypes.get(name, TensorProto.FLOAT), None)
                    value_info.type.tensor_type.shape.dim.add().dim_param = "batch"
                    by_name[name] = value_info
            del graph.graph.input[:]
            for name in expected:
                graph.graph.input.append(by_name[name])

        ensure_contract_inputs(graph)
        entry = graph.metadata_props.add()
        entry.key, entry.value = "rover.format", "rover-policy-onnx-v1"
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
    else:
        export_onnx(str(path), module, obs_dim=module.obs_dim,
                    memory_size=module.memory_size)
    print(f"exported {path} ({path.stat().st_size:,} bytes)", flush=True)


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    returns = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def collect_episodes(env, policy, state, trajectories, episodes, cfg,
                     device, rng, budget, macro, collect_budget, frames,
                     start_time, max_seconds, distances):
    """Step the environment until `collect_budget` decisions are collected.

    Completed trials are discounted and appended to `episodes` (lists of
    per-episode CPU tensors). Finished-trial distances are appended to
    `distances`. Returns the number of collected decisions.
    """
    memory = torch.zeros(env.num_envs, 1, device=device)
    collected = 0
    while collected < collect_budget and time.monotonic() - start_time < max_seconds:
        obs = torch.from_numpy(state["obs"].copy()).to(device)
        with torch.no_grad():
            logits, _, _ = policy.step(
                obs, state["prev_action"], state["prev_reward"], state["prev_done"],
                state["progress"], state["trial_start"], memory,
            )
            action = Categorical(logits=logits).sample()
        action_np = action.cpu().numpy()

        controls = macro[action_np].copy()
        rewards = np.zeros(cfg.num_envs, dtype=np.float32)
        done = np.zeros(cfg.num_envs, dtype=bool)
        frame_counts = np.zeros(cfg.num_envs, dtype=np.int32)
        for _ in range(cfg.frame_skip):
            _, step_reward, terminated, truncated, _ = env.step_uint8(controls)
            active = ~done
            rewards[active] += step_reward[active]
            frame_counts[active] += 1
            state["best"][active] = np.maximum(
                state["best"][active], env.obs[active, 0] * 1000.0)
            done |= terminated.astype(bool) | truncated.astype(bool)
            controls[done] = 0
            if done.all():
                break

        prev_action_np = state["prev_action"].cpu().numpy()
        prev_reward_np = state["prev_reward"].cpu().numpy()
        prev_done_np = state["prev_done"].cpu().numpy()
        progress_np = state["progress"].cpu().numpy()
        trial_start_np = state["trial_start"].cpu().numpy()
        for i in range(cfg.num_envs):
            traj = trajectories[i]
            traj["obs"].append(state["obs"][i].copy())
            traj["prev_action"].append(int(prev_action_np[i]))
            traj["prev_reward"].append(float(prev_reward_np[i]))
            traj["prev_done"].append(float(prev_done_np[i]))
            traj["progress"].append(float(progress_np[i]))
            traj["trial_start"].append(float(trial_start_np[i]))
            traj["action"].append(int(action_np[i]))
            traj["reward"].append(float(rewards[i]))

        state["steps"] += frame_counts
        trial_start = np.zeros(cfg.num_envs, dtype=np.float32)
        for i in np.flatnonzero(done):
            distances.append(max(0.0, float(state["best"][i] - 1.0)))
            state["best"][i] = env.obs[i, 0] * 1000.0
            finalize_episode(trajectories[i], episodes, cfg.gamma)
            trajectories[i] = new_trajectory()
            new_trial = bool(env.next_trial_start[i])
            env.reset_at(int(i), seed=int(rng.integers(0, 2**31)),
                         trial_start=new_trial)
            if new_trial:
                state["steps"][i] = 0
                trial_start[i] = 1.0

        state["obs"] = env.obs.copy()
        state["prev_action"] = action
        state["prev_reward"] = torch.from_numpy(rewards).to(device)
        state["prev_done"] = torch.from_numpy(done.astype(np.float32)).to(device)
        state["progress"] = torch.from_numpy(
            np.minimum(state["steps"] / budget, 1.0).astype(np.float32)
        ).to(device)
        state["trial_start"] = torch.from_numpy(trial_start).to(device)
        frames[0] += cfg.num_envs * cfg.frame_skip
        collected += cfg.num_envs
    return collected


def new_trajectory() -> dict:
    return {
        "obs": [], "prev_action": [], "prev_reward": [], "prev_done": [],
        "progress": [], "trial_start": [], "action": [], "reward": [],
    }


def finalize_episode(traj: dict, episodes: list, gamma: float) -> None:
    if not traj["reward"]:
        return
    episode = {
        "obs": torch.from_numpy(np.stack(traj["obs"])),
        "prev_action": torch.from_numpy(np.asarray(traj["prev_action"], dtype=np.int64)),
        "prev_reward": torch.from_numpy(np.asarray(traj["prev_reward"], dtype=np.float32)),
        "prev_done": torch.from_numpy(np.asarray(traj["prev_done"], dtype=np.float32)),
        "progress": torch.from_numpy(np.asarray(traj["progress"], dtype=np.float32)),
        "trial_start": torch.from_numpy(np.asarray(traj["trial_start"], dtype=np.float32)),
        "action": torch.from_numpy(np.asarray(traj["action"], dtype=np.int64)),
        "returns": torch.from_numpy(discounted_returns(traj["reward"], gamma)),
    }
    episodes.append(episode)


def update_policy(policy, optimizer, episodes, cfg, device):
    if not episodes:
        return None
    policy.train()
    losses = []
    entropy_sum = 0.0
    steps = 0
    for episode in episodes:
        obs = episode["obs"].to(device)
        prev_action = episode["prev_action"].to(device)
        prev_reward = episode["prev_reward"].to(device)
        prev_done = episode["prev_done"].to(device)
        progress = episode["progress"].to(device)
        logits, value = policy.evaluate(obs, prev_action, prev_reward, prev_done, progress)
        distribution = Categorical(logits=logits)
        logprob = distribution.log_prob(episode["action"].to(device))
        returns = episode["returns"].to(device)
        advantage = returns - value.detach()
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8)
        actor_loss = -(logprob * advantage).mean()
        critic_loss = 0.5 * (value - returns).square().mean()
        entropy = distribution.entropy().mean()
        losses.append(actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy)
        entropy_sum += float(entropy)
        steps += len(episode["returns"])
    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    optimizer.step()
    return float(total_loss), entropy_sum / max(1, len(episodes)), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-frames", type=int, default=int(
        os.environ.get("TOTAL_FRAMES", "12000000")))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--seed", type=int, default=int(
        os.environ.get("ARENA_SEED", "2026")))
    parser.add_argument("--max-seconds", type=int, default=6900)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--update-every", type=int, default=32768)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    default_output = Path("/output/policy.onnx") if Path("/output").is_dir() else (
        Path(__file__).resolve().parent / "artifacts" / "policy.onnx"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--checkpoint", type=Path, default=(
        Path("/tmp/training/latest.pt") if Path("/output").is_dir() else
        Path(__file__).resolve().parent / "artifacts" / "latest.pt"
    ))
    args = parser.parse_args()
    if min(args.num_envs, args.frame_skip, args.total_frames, args.max_seconds,
           args.update_every) <= 0:
        parser.error("all counts and durations must be positive")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MarsRoverVecEnv(args.num_envs)
    if env.obs_dim != 160 or len(ACTION_MACROS) != 31:
        raise RuntimeError("The evaluator observation/action contract changed")
    policy = Reinforce(env.obs_dim, len(ACTION_MACROS), 128).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    zeros = lambda: torch.zeros(args.num_envs, device=device)
    obs = env.reset(args.seed).copy()
    state = {
        "obs": obs,
        "prev_action": torch.zeros(args.num_envs, dtype=torch.long, device=device),
        "prev_reward": zeros(),
        "prev_done": zeros(),
        "progress": zeros(),
        "trial_start": torch.ones(args.num_envs, device=device),
        "steps": np.zeros(args.num_envs, dtype=np.int32),
        "best": obs[:, 0].copy() * 1000.0,
    }
    cfg = argparse.Namespace(
        num_envs=args.num_envs, frame_skip=args.frame_skip, gamma=args.gamma,
        value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        max_grad_norm=0.5,
    )
    trajectories = [new_trajectory() for _ in range(args.num_envs)]
    episodes = []
    macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    budget = max(1, round(env.trial_time_limit * 60))
    start = time.monotonic()
    frames = [0]
    updates = 0
    distances = []
    print(f"training on {device}", flush=True)
    export_policy(policy, args.output)
    while frames[0] < args.total_frames and time.monotonic() - start < args.max_seconds:
        collect_episodes(
            env, policy, state, trajectories, episodes, cfg, device, rng,
            budget, macro, args.update_every, frames, start, args.max_seconds,
            distances,
        )
        result = update_policy(policy, optimizer, episodes, cfg, device)
        episodes.clear()
        if result is None:
            continue
        total_loss, entropy, steps = result
        updates += 1
        median = np.median(distances) if distances else float("nan")
        distances.clear()
        if updates % args.save_every == 0 or frames[0] >= args.total_frames:
            export_policy(policy, args.output)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": policy.state_dict(), "frames": frames[0]},
                       args.checkpoint)
        print(
            f"update={updates} frames={frames[0]} steps={steps} loss={total_loss:.3f} "
            f"entropy={entropy:.3f} median_distance={median:.1f}m "
            f"elapsed={time.monotonic() - start:.0f}s",
            flush=True,
        )
    export_policy(policy, args.output)


if __name__ == "__main__":
    main()
