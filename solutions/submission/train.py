"""Train a recurrent PPO rover policy with the public vector simulator."""

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
from model import Policy


def export_policy(model: Policy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = copy.deepcopy(model).cpu().eval()
    try:
        from arena.protocol import export_onnx
    except ModuleNotFoundError as exc:
        if exc.name != "arena":
            raise
        # The local environment lacks arena-base. The server uses its validator.
        import onnx

        n = 2
        example = (
            torch.zeros(n, module.obs_dim), torch.zeros(n, dtype=torch.int64),
            torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.ones(n),
            torch.zeros(n, module.hidden_size),
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
        entry = graph.metadata_props.add()
        entry.key, entry.value = "rover.format", "rover-policy-onnx-v1"
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
    else:
        export_onnx(str(path), module, obs_dim=module.obs_dim,
                    memory_size=module.hidden_size)
    print(f"exported {path} ({path.stat().st_size:,} bytes)", flush=True)


def collect(env, policy, state, cfg, device, rng):
    keys = (
        "obs", "prev_action", "prev_reward", "prev_done", "progress",
        "trial_start", "action", "logprob", "value", "reward", "done",
    )
    data = {key: [] for key in keys}
    start_memory = state["memory"].detach().clone()
    macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    finished_distances = []
    budget = max(1, round(env.trial_time_limit * 60))
    policy.eval()

    for _ in range(cfg.rollout_steps):
        obs = torch.from_numpy(state["obs"].copy()).to(device)
        features = (
            obs, state["prev_action"], state["prev_reward"],
            state["prev_done"], state["progress"], state["trial_start"],
        )
        with torch.no_grad():
            logits, value, next_memory = policy.step(*features, state["memory"])
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            logprob = distribution.log_prob(action)
        for key, tensor in zip(keys, (*features, action, logprob, value)):
            data[key].append(tensor)

        controls = macro[action.cpu().numpy()].copy()
        rewards = np.zeros(cfg.num_envs, dtype=np.float32)
        done = np.zeros(cfg.num_envs, dtype=bool)
        frame_counts = np.zeros(cfg.num_envs, dtype=np.int32)
        for _ in range(cfg.frame_skip):
            _, step_reward, terminated, truncated, _ = env.step_uint8(controls)
            active = ~done
            rewards[active] += step_reward[active]
            frame_counts[active] += 1
            state["best"][active] = np.maximum(
                state["best"][active], env.obs[active, 0] * 1000.0
            )
            done |= terminated.astype(bool) | truncated.astype(bool)
            controls[done] = 0
            if done.all():
                break

        state["steps"] += frame_counts
        trial_start = np.zeros(cfg.num_envs, dtype=np.float32)
        for i in np.flatnonzero(done):
            finished_distances.append(max(0.0, float(state["best"][i] - 1.0)))
            new_trial = bool(env.next_trial_start[i])
            env.reset_at(int(i), seed=int(rng.integers(0, 2**31)),
                         trial_start=new_trial)
            if new_trial:
                state["steps"][i] = 0
                trial_start[i] = 1.0
            state["best"][i] = env.obs[i, 0] * 1000.0

        data["reward"].append(torch.from_numpy(rewards).to(device))
        data["done"].append(torch.from_numpy(done.astype(np.float32)).to(device))
        state["obs"] = env.obs.copy()
        state["prev_action"] = action
        state["prev_reward"] = torch.from_numpy(rewards).to(device)
        state["prev_done"] = torch.from_numpy(done.astype(np.float32)).to(device)
        state["progress"] = torch.from_numpy(
            np.minimum(state["steps"] / budget, 1.0).astype(np.float32)
        ).to(device)
        state["trial_start"] = torch.from_numpy(trial_start).to(device)
        state["memory"] = next_memory

    with torch.no_grad():
        _, bootstrap, _ = policy.step(
            torch.from_numpy(state["obs"]).to(device), state["prev_action"],
            state["prev_reward"], state["prev_done"], state["progress"],
            state["trial_start"], state["memory"],
        )
    batch = {key: torch.stack(value) for key, value in data.items()}
    batch["start_memory"] = start_memory
    return batch, bootstrap, finished_distances


def update_policy(policy, optimizer, batch, bootstrap, cfg, rng):
    reward, done, value = batch["reward"], batch["done"], batch["value"]
    advantage = torch.empty_like(reward)
    gae = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(cfg.rollout_steps)):
        alive = 1.0 - done[t]
        delta = reward[t] + cfg.gamma * next_value * alive - value[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * alive * gae
        advantage[t] = gae
        next_value = value[t]
    returns = advantage + value
    advantage = (advantage - advantage.mean()) / (
        advantage.std(unbiased=False) + 1e-8
    )
    policy.train()
    metrics = []
    chunks = max(1, (cfg.num_envs + cfg.envs_per_batch - 1) // cfg.envs_per_batch)
    for _ in range(cfg.epochs):
        for indices in np.array_split(rng.permutation(cfg.num_envs), chunks):
            ids = indices.tolist()
            logits, predicted_value = policy.sequence(
                *(batch[key][:, ids] for key in (
                    "obs", "prev_action", "prev_reward", "prev_done",
                    "progress", "trial_start",
                )),
                batch["start_memory"][ids],
            )
            distribution = Categorical(logits=logits)
            logprob = distribution.log_prob(batch["action"][:, ids])
            ratio = (logprob - batch["logprob"][:, ids]).exp()
            selected_adv = advantage[:, ids]
            actor_loss = -torch.minimum(
                ratio * selected_adv,
                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * selected_adv,
            ).mean()
            critic_loss = 0.5 * (
                predicted_value - returns[:, ids]
            ).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            metrics.append((actor_loss.item(), critic_loss.item(), entropy.item()))
    return np.mean(metrics, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-frames", type=int, default=int(
        os.environ.get("TOTAL_FRAMES", "12000000")))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=int(
        os.environ.get("ARENA_SEED", "2026")))
    parser.add_argument("--max-seconds", type=int, default=6900)
    parser.add_argument("--save-every", type=int, default=10)
    default_output = Path("/output/policy.onnx") if Path("/output").is_dir() else (
        Path(__file__).resolve().parent / "artifacts" / "policy.onnx"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--checkpoint", type=Path, default=(
        Path("/tmp/training/latest.pt") if Path("/output").is_dir() else
        Path(__file__).resolve().parent / "artifacts" / "latest.pt"
    ))
    args = parser.parse_args()
    if min(args.num_envs, args.rollout_steps, args.frame_skip, args.epochs,
           args.total_frames, args.max_seconds, args.save_every) <= 0:
        parser.error("all counts and durations must be positive")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MarsRoverVecEnv(args.num_envs)
    if env.obs_dim != 160 or len(ACTION_MACROS) != 31:
        raise RuntimeError("The evaluator observation/action contract changed")
    policy = Policy(env.obs_dim, len(ACTION_MACROS), 128).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    zeros = lambda: torch.zeros(args.num_envs, device=device)
    obs = env.reset(args.seed).copy()
    state = {
        "obs": obs,
        "prev_action": torch.zeros(args.num_envs, dtype=torch.long, device=device),
        "prev_reward": zeros(),
        "prev_done": zeros(),
        "progress": zeros(),
        "trial_start": torch.ones(args.num_envs, device=device),
        "memory": torch.zeros(args.num_envs, 128, device=device),
        "steps": np.zeros(args.num_envs, dtype=np.int32),
        "best": obs[:, 0].copy() * 1000.0,
    }
    cfg = argparse.Namespace(
        num_envs=args.num_envs, rollout_steps=args.rollout_steps,
        frame_skip=args.frame_skip, epochs=args.epochs, envs_per_batch=16,
        gamma=0.995, gae_lambda=0.95, clip=0.2, value_coef=0.5,
        entropy_coef=0.01, max_grad_norm=0.5,
    )
    start = time.monotonic()
    frames = 0
    update = 0
    print(f"training on {device}", flush=True)
    export_policy(policy, args.output)
    while frames < args.total_frames and time.monotonic() - start < args.max_seconds:
        batch, bootstrap, distances = collect(env, policy, state, cfg, device, rng)
        actor, critic, entropy = update_policy(
            policy, optimizer, batch, bootstrap, cfg, rng
        )
        frames += args.num_envs * args.rollout_steps * args.frame_skip
        update += 1
        if update % args.save_every == 0 or frames >= args.total_frames:
            export_policy(policy, args.output)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                 "frames": frames}, args.checkpoint,
            )
        median = np.median(distances) if distances else float("nan")
        print(
            f"update={update} frames={frames} episodes={len(distances)} "
            f"median_distance={median:.1f}m actor={actor:.3f} "
            f"critic={critic:.3f} entropy={entropy:.3f} "
            f"elapsed={time.monotonic() - start:.0f}s",
            flush=True,
        )
    export_policy(policy, args.output)


if __name__ == "__main__":
    main()
