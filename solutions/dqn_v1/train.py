"""Train a Double DQN rover policy with the public vector simulator."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from model import DQN


def export_policy(model: DQN, path: Path) -> None:
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


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.action_dim = action_dim
        self.size = 0
        self.pos = 0
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.prev_action = np.empty((capacity,), dtype=np.int64)
        self.prev_reward = np.empty((capacity,), dtype=np.float32)
        self.prev_done = np.empty((capacity,), dtype=np.float32)
        self.progress = np.empty((capacity,), dtype=np.float32)
        self.action = np.empty((capacity,), dtype=np.int64)
        self.reward = np.empty((capacity,), dtype=np.float32)
        self.done = np.empty((capacity,), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_prev_action = np.empty((capacity,), dtype=np.int64)
        self.next_prev_reward = np.empty((capacity,), dtype=np.float32)
        self.next_prev_done = np.empty((capacity,), dtype=np.float32)
        self.next_progress = np.empty((capacity,), dtype=np.float32)

    def push(self, obs, prev_action, prev_reward, prev_done, progress,
             action, reward, done, next_obs, next_prev_action,
             next_prev_reward, next_prev_done, next_progress):
        idx = self.pos
        self.obs[idx] = obs
        self.prev_action[idx] = prev_action
        self.prev_reward[idx] = prev_reward
        self.prev_done[idx] = prev_done
        self.progress[idx] = progress
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        self.next_obs[idx] = next_obs
        self.next_prev_action[idx] = next_prev_action
        self.next_prev_reward[idx] = next_prev_reward
        self.next_prev_done[idx] = next_prev_done
        self.next_progress[idx] = next_progress
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator):
        indices = rng.choice(self.size, batch_size, replace=False)
        arrays = (
            self.obs, self.prev_action, self.prev_reward, self.prev_done,
            self.progress, self.action, self.reward, self.done, self.next_obs,
            self.next_prev_action, self.next_prev_reward, self.next_prev_done,
            self.next_progress,
        )
        return tuple(torch.from_numpy(array[indices]).to(device) for array in arrays)


def select_actions(policy, obs, ctx, epsilon, action_dim, rng):
    with torch.no_grad():
        q = policy.q_values(
            obs, ctx["prev_action"], ctx["prev_reward"], ctx["prev_done"], ctx["progress"],
        )
    greedy = q.argmax(1).cpu().numpy()
    random = rng.random(obs.shape[0]) < epsilon
    rand_actions = rng.integers(0, action_dim, size=obs.shape[0])
    return np.where(random, rand_actions, greedy)


def update_dqn(online, target, optimizer, buffer, batch_size, gamma, tau,
               device, rng):
    batch = buffer.sample(batch_size, device, rng)
    (obs, prev_action, prev_reward, prev_done, progress, action, reward,
     done, next_obs, next_prev_action, next_prev_reward, next_prev_done,
     next_progress) = batch

    q_sa = online.q_values(obs, prev_action, prev_reward, prev_done, progress).gather(
        1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_argmax = online.q_values(
            next_obs, next_prev_action, next_prev_reward, next_prev_done,
            next_progress).argmax(1)
        next_value = target.q_values(
            next_obs, next_prev_action, next_prev_reward, next_prev_done,
            next_progress).gather(1, next_argmax.unsqueeze(1)).squeeze(1)
        target_value = reward + gamma * (1.0 - done) * next_value
    loss = nn.functional.smooth_l1_loss(q_sa, target_value)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    with torch.no_grad():
        for parameter, target_parameter in zip(online.parameters(), target.parameters()):
            target_parameter.lerp_(parameter, tau)
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-frames", type=int, default=int(
        os.environ.get("TOTAL_FRAMES", "12000000")))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--seed", type=int, default=int(
        os.environ.get("ARENA_SEED", "2026")))
    parser.add_argument("--max-seconds", type=int, default=6900)
    parser.add_argument("--save-every", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
           args.batch_size, args.capacity) <= 0:
        parser.error("all counts and durations must be positive")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MarsRoverVecEnv(args.num_envs)
    action_dim = len(ACTION_MACROS)
    if env.obs_dim != 160 or action_dim != 31:
        raise RuntimeError("The evaluator observation/action contract changed")
    macro = np.asarray(ACTION_MACROS, dtype=np.int32)

    online = DQN(env.obs_dim, action_dim).to(device)
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate, eps=1e-5)
    buffer = ReplayBuffer(args.capacity, env.obs_dim, action_dim)

    zeros = lambda: torch.zeros(args.num_envs, device=device)
    obs = env.reset(args.seed).copy()
    ctx = {
        "obs": torch.from_numpy(obs).to(device),
        "prev_action": torch.zeros(args.num_envs, dtype=torch.long, device=device),
        "prev_reward": zeros(),
        "prev_done": zeros(),
        "progress": zeros(),
    }
    state = {
        "steps": np.zeros(args.num_envs, dtype=np.int32),
        "best": obs[:, 0].copy() * 1000.0,
    }
    budget = max(1, round(env.trial_time_limit * 60))
    start = time.monotonic()
    frames = 0
    updates = 0
    last_save = 0
    print(f"training on {device}", flush=True)
    export_policy(online, args.output)
    while frames < args.total_frames and time.monotonic() - start < args.max_seconds:
        epsilon = max(
            args.eps_end,
            args.eps_start - (args.eps_start - args.eps_end) * frames / args.total_frames,
        )
        finished_distances = []
        for _ in range(1024):
            actions = select_actions(online, ctx["obs"], ctx, epsilon, action_dim, rng)
            controls = macro[actions].copy()
            rewards = np.zeros(args.num_envs, dtype=np.float32)
            done = np.zeros(args.num_envs, dtype=bool)
            frame_counts = np.zeros(args.num_envs, dtype=np.int32)
            for _ in range(args.frame_skip):
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

            state["steps"] += frame_counts
            next_obs = env.obs.copy()
            next_progress = np.minimum(state["steps"] / budget, 1.0).astype(np.float32)
            next_prev_action = actions.astype(np.int64)
            next_prev_reward = rewards.astype(np.float32)
            next_prev_done = done.astype(np.float32)
            for i in np.flatnonzero(done):
                finished_distances.append(max(0.0, float(state["best"][i] - 1.0)))
                new_trial = bool(env.next_trial_start[i])
                env.reset_at(int(i), seed=int(rng.integers(0, 2**31)),
                             trial_start=new_trial)
                if new_trial:
                    state["steps"][i] = 0
                state["best"][i] = env.obs[i, 0] * 1000.0
                next_obs[i] = env.obs[i]
                next_progress[i] = min(state["steps"][i] / budget, 1.0)
                next_prev_action[i] = 0
                next_prev_reward[i] = 0.0
                next_prev_done[i] = 0.0

            obs_np = ctx["obs"].cpu().numpy()
            prev_action_np = ctx["prev_action"].cpu().numpy()
            prev_reward_np = ctx["prev_reward"].cpu().numpy()
            prev_done_np = ctx["prev_done"].cpu().numpy()
            progress_np = ctx["progress"].cpu().numpy()
            for i in range(args.num_envs):
                buffer.push(
                    obs_np[i], prev_action_np[i], prev_reward_np[i],
                    prev_done_np[i], progress_np[i], actions[i], rewards[i],
                    done[i], next_obs[i], next_prev_action[i],
                    next_prev_reward[i], next_prev_done[i], next_progress[i],
                )
            ctx = {
                "obs": torch.from_numpy(next_obs).to(device),
                "prev_action": torch.from_numpy(next_prev_action).to(device),
                "prev_reward": torch.from_numpy(next_prev_reward).to(device),
                "prev_done": torch.from_numpy(next_prev_done).to(device),
                "progress": torch.from_numpy(next_progress).to(device),
            }
            frames += args.num_envs * args.frame_skip
            if buffer.size >= max(args.batch_size, 1000):
                loss = update_dqn(
                    online, target, optimizer, buffer, args.batch_size,
                    args.gamma, args.tau, device, rng,
                )
                updates += 1
            if time.monotonic() - start >= args.max_seconds:
                break

        if frames - last_save >= args.save_every:
            last_save = frames
            export_policy(online, args.output)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": online.state_dict(), "frames": frames},
                       args.checkpoint)
        median = np.median(finished_distances) if finished_distances else float("nan")
        print(
            f"frames={frames} eps={epsilon:.3f} updates={updates} "
            f"episodes={len(finished_distances)} median_distance={median:.1f}m "
            f"buffer={buffer.size} elapsed={time.monotonic() - start:.0f}s",
            flush=True,
        )
    export_policy(online, args.output)


if __name__ == "__main__":
    main()
