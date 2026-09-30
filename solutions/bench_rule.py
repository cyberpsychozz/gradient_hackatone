"""Bench a torch RulePolicy on regular/dense/weather configs (median distance)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "rover_best_v1"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mars_rover_env import MarsRoverVecEnv  # noqa: E402
from mars_rover_env.actions import ACTION_MACROS  # noqa: E402
from rule_policy import RulePolicy  # noqa: E402


def run(policy, config_path, num_envs, base_seed, max_actions=2250, frame_skip=8):
    env = MarsRoverVecEnv(num_envs, config_path=str(config_path))
    macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    memory = torch.zeros(num_envs, 1)
    best_x = np.full(num_envs, 1.0, dtype=np.float32)
    done_flags = np.zeros(num_envs, dtype=bool)
    env.reset(base_seed)
    for i in range(1, num_envs):
        env.reset_at(i, seed=base_seed + i)
    best_x[:] = env.obs[:, 0] * 1000.0
    prev_action = torch.zeros(num_envs, dtype=torch.long)
    prev_reward = torch.zeros(num_envs)
    prev_done = torch.zeros(num_envs)
    progress = torch.zeros(num_envs)
    trial_start = torch.ones(num_envs)
    steps = np.zeros(num_envs, dtype=np.int32)
    with torch.no_grad():
        for _ in range(max_actions):
            obs = torch.from_numpy(env.obs.copy())
            logits, next_memory = policy(
                obs, prev_action, prev_reward, prev_done, progress, trial_start, memory)
            memory = next_memory
            action = logits.argmax(-1)
            controls = macro[action.numpy()].copy()
            for _ in range(frame_skip):
                _, _, term, trunc, _ = env.step_uint8(controls)
                np.maximum(best_x, env.obs[:, 0] * 1000.0, out=best_x)
                steps += 1
                done_flags |= term.astype(bool) | trunc.astype(bool)
                controls[done_flags] = 0
            progress = torch.from_numpy(
                np.minimum(steps / 18000.0, 1.0).astype(np.float32))
            prev_action = action
            prev_reward = torch.from_numpy(env.rewards.copy())
            prev_done = torch.from_numpy(done_flags.astype(np.float32))
            trial_start = torch.zeros(num_envs)
            if done_flags.all():
                break
    return np.maximum(0.0, best_x - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="904,1904,2904")
    ap.add_argument("--configs", type=str,
                    default="python/mars_rover_env/configs/env.yaml,"
                            "python/mars_rover_env/configs/eval_stress_dense.yaml,"
                            "python/mars_rover_env/configs/eval_stress_weather.yaml")
    ap.add_argument("--drop", type=float)
    ap.add_argument("--rise", type=float)
    ap.add_argument("--slope", type=float)
    ap.add_argument("--lookahead", type=float)
    ap.add_argument("--brake-speed", type=float)
    ap.add_argument("--lidar", action="store_true")
    args = ap.parse_args()
    if args.drop is not None:
        RulePolicy.drop_threshold = args.drop
    if args.rise is not None:
        RulePolicy.rise_threshold = args.rise
    if args.slope is not None:
        RulePolicy.slope_threshold = args.slope
    if args.lookahead is not None:
        RulePolicy.lookahead = args.lookahead
    if args.brake_speed is not None:
        RulePolicy.brake_speed = args.brake_speed
    if not args.lidar:
        RulePolicy._lidar_enabled = False
    policy = RulePolicy()
    policy.eval()
    for cfg in args.configs.split(","):
        medians = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            d = run(policy, ROOT / cfg, args.envs, seed)
            medians.append(float(np.median(d)))
        print(f"{Path(cfg).name}: medians={[round(m,1) for m in medians]} "
              f"overall={round(float(np.median(medians)),1)}m")


if __name__ == "__main__":
    main()
