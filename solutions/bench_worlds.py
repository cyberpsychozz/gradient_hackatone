#!/usr/bin/env python3
"""Benchmark exported ONNX rover policies on fixed worlds and seeds.

Usage:
    python bench_worlds.py POLICY.onnx [POLICY2.onnx ...] --out report.md

Evaluates each policy on every registered biome (fixed_biome_id, chain off)
plus the default biome chain, with fixed trial seeds and full 300s trials
(18000 physics steps). Per world the score is the median of per-trial max
distance minus the first metre.

Caveat: in this local build, biome callbacks (apply_effects/apply_body_effects)
are active only for biome IDs 20-39; IDs 0-19 reflect base physics only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from _mars_rover_cpp import biome_catalog
from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import load_env_config

FRAME_SKIP = 8
WORLD_SEEDS = (42, 2026, 7, 1337)
CHAIN_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)


def load_policy(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    memory_size = session.get_inputs()[-1].shape[1]
    return session, memory_size


def evaluate(session, memory_size, env, budget, macro, seeds):
    num_envs = len(seeds)
    for env_id, seed in enumerate(seeds):
        env.reset_at(env_id, seed=int(seed), trial_start=True)
    obs = env.obs.copy().astype(np.float32)
    prev_a = np.zeros(num_envs, dtype=np.int64)
    prev_r = np.zeros(num_envs, dtype=np.float32)
    prev_d = np.zeros(num_envs, dtype=np.float32)
    prog = np.zeros(num_envs, dtype=np.float32)
    ts = np.ones(num_envs, dtype=np.float32)
    mem = np.zeros((num_envs, memory_size), dtype=np.float32)
    best = obs[:, 0].copy() * 1000.0
    steps = np.zeros(num_envs, dtype=np.int32)
    done = np.zeros(num_envs, dtype=bool)

    for _ in range(budget // FRAME_SKIP):
        logits = session.run(None, {
            "observation": obs,
            "previous_action": prev_a,
            "previous_reward": prev_r,
            "previous_done": prev_d,
            "trial_progress": prog,
            "trial_start": ts,
            "memory": mem,
        })[0]
        action = np.argmax(logits, axis=1).astype(np.int64)
        controls = macro[action].copy()
        controls[done] = 0
        rewards = np.zeros(num_envs, dtype=np.float32)
        decision_done = np.zeros(num_envs, dtype=bool)
        for _ in range(FRAME_SKIP):
            _, reward, terminated, truncated, _ = env.step_uint8(controls)
            active = ~done & ~decision_done
            rewards[active] += reward[active]
            steps[active] += 1
            best[active] = np.maximum(best[active], env.obs[active, 0] * 1000.0)
            decision_done |= terminated.astype(bool) | truncated.astype(bool)
            done |= decision_done
            controls[done] = 0
            if done.all():
                break
        prog = np.minimum(steps / budget, 1.0).astype(np.float32)
        prev_a = action
        prev_r = rewards
        prev_d = done.astype(np.float32)
        obs = env.obs.copy().astype(np.float32)
    return np.maximum(best - 1.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="+", type=Path, help="ONNX policy files")
    parser.add_argument("--out", type=Path, default=Path("report.md"))
    args = parser.parse_args()

    macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    catalog = biome_catalog()
    budget = round(load_env_config().termination.trial_time_limit * 60)

    results = {}
    for policy_path in args.policies:
        if not policy_path.is_file():
            raise SystemExit(f"policy not found: {policy_path}")
        session, memory_size = load_policy(policy_path)
        started = time.monotonic()
        per_world = {}

        for biome_id, world in enumerate(catalog):
            env = MarsRoverVecEnv(len(WORLD_SEEDS), fixed_biome_id=biome_id)
            distances = evaluate(
                session, memory_size, env, budget, macro, WORLD_SEEDS)
            per_world[f"{biome_id}:{world['id']}"] = {
                "median": float(np.median(distances)),
                "max": float(distances.max()),
                "seeds": [float(d) for d in distances],
            }

        chain_env = MarsRoverVecEnv(len(CHAIN_SEEDS))
        chain = evaluate(
            session, memory_size, chain_env, budget, macro, CHAIN_SEEDS)
        per_world["chain"] = {
            "median": float(np.median(chain)),
            "max": float(chain.max()),
            "seeds": [float(d) for d in chain],
        }
        results[str(policy_path)] = per_world
        print(f"{policy_path}: {len(per_world)} worlds in "
              f"{time.monotonic() - started:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
