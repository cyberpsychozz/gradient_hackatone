"""Measure the deterministic controller on every training-world archetype.

This is a local calibration tool. It reports the platform-like median maximum
progress per 300-second trial, using different deterministic terrain seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from rule_policy import RulePolicy
from train import quick_eval
from world_profiles import PROFILES, allocate_profiles, training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=84000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    torch.set_num_threads(2)
    policy = RulePolicy()
    macros = np.asarray(ACTION_MACROS, dtype=np.int32)
    allocation = {profile.name: count for profile, count in allocate_profiles(64)}
    rows = []
    for profile in PROFILES:
        env = MarsRoverVecEnv(args.runs, config_override=training_config(profile))
        stats = quick_eval(policy, env, macros, args.seed, torch.device("cpu"),
                           policy.hidden_size)
        row = {"name": profile.name, "tier": profile.tier,
               "train_envs_of_64": allocation[profile.name],
               "median_m": round(stats["median"], 1),
               "mean_m": round(stats["mean"], 1)}
        rows.append(row)
        print(f"{profile.name:20s} {profile.tier:11s} "
              f"{allocation[profile.name]:2d}/64 "
              f"median={stats['median']:7.1f} m "
              f"mean={stats['mean']:7.1f} m", flush=True)
    result = {"runs_per_profile": args.runs, "seed": args.seed,
              "action_policy": "rule_policy", "profiles": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
