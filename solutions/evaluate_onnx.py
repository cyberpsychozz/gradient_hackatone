"""Evaluate ONNX rover policy (or a constant action) on independent local trials.

Use an installed mars_rover_env matching the intended biome bank on PYTHONPATH.
This reproduces the platform's eight-physics-frame action and 300 s budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS


def evaluate(args):
    env = MarsRoverVecEnv(args.runs, config_path=args.config or None,
                          biome_split=args.biome_split,
                          fixed_biome_id=args.biome_id)
    env.reset(args.seed)
    for i in range(args.runs):
        env.reset_at(i, seed=args.seed + i * 7919)
    obs = env.obs.copy()
    memory = np.zeros((args.runs, 1), dtype=np.float32)
    session = None
    if args.policy:
        import onnxruntime as ort
        session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"],
                                       sess_options=ort.SessionOptions())
        memory = np.zeros((args.runs, session.get_inputs()[-1].shape[1]), np.float32)
    macro = np.asarray(ACTION_MACROS, np.int32)
    best = obs[:, 0].copy() * 1000
    done = np.zeros(args.runs, bool)
    energy_at_best = obs[:, 6].copy()
    step_at_best = np.zeros(args.runs, np.int32)
    steps = np.zeros(args.runs, np.int32)
    prev_action = np.zeros(args.runs, np.int64)
    prev_reward = np.zeros(args.runs, np.float32)
    prev_done = np.zeros(args.runs, np.float32)
    trial_start = np.ones(args.runs, np.float32)
    counts = np.zeros(len(macro), np.int64)
    run_counts = np.zeros((args.runs, len(macro)), np.int64) if args.trace_actions else None
    counts_at_best = np.zeros_like(run_counts) if args.trace_actions else None
    counts_first_120s = np.zeros(len(macro), np.int64) if args.trace_actions else None
    panel_on = np.zeros(args.runs, bool)
    panel_age = np.zeros(args.runs, np.int32)
    panel_cycles = np.zeros(args.runs, np.int32)
    budget = int(round(env.trial_time_limit * 60))
    max_actions = min(args.max_actions, (budget + args.frame_skip - 1) // args.frame_skip)
    for decision_index in range(max_actions):
        if done.all():
            break
        if session:
            feeds = dict(observation=obs, previous_action=prev_action,
                         previous_reward=prev_reward, previous_done=prev_done,
                         trial_progress=np.minimum(steps / budget, 1).astype(np.float32),
                         trial_start=trial_start, memory=memory)
            logits, memory = session.run(None, feeds)
            if not np.isfinite(logits).all() or not np.isfinite(memory).all():
                raise RuntimeError("Nonfinite policy output")
            action = logits.argmax(axis=1).astype(np.int64)
        elif args.auto_shift:
            action = np.full(args.runs, args.after_action, dtype=np.int64)
            gear = np.rint(obs[:, 104] * 8).astype(np.int32) - 1
            shift = (obs[:, 107] >= args.shift_threshold) & (gear < 5) & (prev_action != 5)
            action[shift] = 5
            if args.prefix_actions and decision_index < len(args.prefix_actions):
                action[:] = args.prefix_actions[decision_index]
            elif args.prefix_actions:
                angle = obs[:, 4] + args.level_damping * obs[:, 5] * 10
                allowed = (obs[:, 151] > 0.5) if args.level_air_only else np.ones(args.runs, bool)
                if args.level_threshold < 10:
                    action[(angle > args.level_threshold) & allowed] = 11
                    action[(angle < -args.level_threshold) & allowed] = 10
            elif args.drive_mode and counts.sum() == 0:
                action[:] = 13
            elif args.drive_mode == 2 and counts.sum() == args.runs:
                action[:] = 1
            elif args.drive_mode == 2 and counts.sum() == 2 * args.runs:
                action[:] = 13
            elif args.level_threshold < 10:
                angle = obs[:, 4] + args.level_damping * obs[:, 5] * 10
                allowed = (obs[:, 151] > 0.5) if args.level_air_only else np.ones(args.runs, bool)
                action[(angle > args.level_threshold) & allowed] = 11
                action[(angle < -args.level_threshold) & allowed] = 10
        else:
            action = np.full(args.runs, args.constant_action, np.int64)
        if args.auto_shift and args.solar_rescue_threshold > 0:
            start_charge = (~done & ~panel_on
                            & (panel_cycles < args.solar_max_cycles)
                            & (obs[:, 6] < args.solar_rescue_threshold)
                            & (np.abs(obs[:, 2] * 20) < args.solar_max_speed)
                            & (decision_index >= max(3, len(args.prefix_actions))))
            panel_on[start_charge] = True
            panel_cycles[start_charge] += 1
            panel_age[start_charge] = 0
            action[start_charge] = 14
            charging = panel_on & ~start_charge & ~done
            panel_age[charging] += 1
            finish_charge = charging & (
                (obs[:, 6] >= args.solar_charge_target)
                | (panel_age >= args.solar_max_actions)
            )
            action[charging] = 0
            action[finish_charge] = 14
            panel_on[finish_charge] = False
        if args.trace_actions:
            active_ids = np.flatnonzero(~done)
            run_counts[active_ids, action[active_ids]] += 1
            first_120 = (~done) & (steps < 120 * 60)
            np.add.at(counts_first_120s, action[first_120], 1)
        np.add.at(counts, action[~done], 1)
        controls = macro[action].copy()
        controls[done] = 0
        rewards = np.zeros(args.runs, np.float32)
        old_done = done.copy()
        improved_any = np.zeros(args.runs, bool) if args.trace_actions else None
        for _ in range(args.frame_skip):
            active = ~done
            if not active.any():
                break
            obs, reward, terminated, truncated, _ = env.step_uint8(controls)
            rewards[active] += reward[active]
            steps[active] += 1
            x = obs[:, 0] * 1000
            improved = active & (x > best)
            if args.trace_actions:
                improved_any |= improved
            energy_at_best[improved] = obs[improved, 6]
            step_at_best[improved] = steps[improved]
            np.maximum(best, x, out=best, where=active)
            done |= terminated.astype(bool) | truncated.astype(bool)
            controls[done] = 0
        if args.trace_actions:
            counts_at_best[improved_any] = run_counts[improved_any]
        prev_action = action
        prev_reward = rewards
        prev_done = old_done.astype(np.float32)
        trial_start.fill(0)
        obs = env.obs.copy()
    distances = np.maximum(0, best - 1)
    result = {
        "policy": str(args.policy) if args.policy else (f"auto_shift:{args.shift_threshold}:drive{args.drive_mode}" if args.auto_shift else f"constant:{args.constant_action}"),
        "runs": args.runs, "seed": args.seed, "biome_split": args.biome_split,
        "biome_id": args.biome_id, "actions": int(counts.sum()),
        "prefix_actions": args.prefix_actions,
        "solar_rescue_threshold": args.solar_rescue_threshold,
        "solar_cycles": int(panel_cycles.sum()),
        "solar_max_speed": (float(args.solar_max_speed)
                            if np.isfinite(args.solar_max_speed) else None),
        "median_m": float(np.median(distances)), "mean_m": float(np.mean(distances)),
        "q25_m": float(np.quantile(distances, .25)),
        "q75_m": float(np.quantile(distances, .75)),
        "distances_m": [round(float(x), 2) for x in distances],
        "action_counts": {str(i): int(n) for i, n in enumerate(counts) if n},
        "finished": int(done.sum()),
        "median_energy_at_best": float(np.median(energy_at_best)),
        "median_time_at_best_s": float(np.median(step_at_best) / 60),
    }
    if args.trace_actions:
        before_best = counts_at_best.sum(axis=0)
        after_best = counts - before_best
        result["action_timing"] = {
            "first_120_s": {str(i): int(n) for i, n in enumerate(counts_first_120s) if n},
            "until_best": {str(i): int(n) for i, n in enumerate(before_best) if n},
            "after_best": {str(i): int(n) for i, n in enumerate(after_best) if n},
        }
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--constant-action", type=int, default=1)
    parser.add_argument("--auto-shift", action="store_true")
    parser.add_argument("--after-action", type=int, default=1)
    parser.add_argument("--shift-threshold", type=float, default=0.78)
    parser.add_argument("--drive-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prefix-actions", type=str, default="",
                        help="comma-separated macros used once at the start of each trial")
    parser.add_argument("--level-threshold", type=float, default=10.0)
    parser.add_argument("--level-damping", type=float, default=0.0)
    parser.add_argument("--level-air-only", action="store_true")
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=904)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--biome-split", type=int, default=None)
    parser.add_argument("--biome-id", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=2250)
    parser.add_argument("--trace-actions", action="store_true",
                        help="count macros in the first 120 s and before/after each run's best distance")
    parser.add_argument("--solar-rescue-threshold", type=float, default=0.0)
    parser.add_argument("--solar-charge-target", type=float, default=0.8)
    parser.add_argument("--solar-max-actions", type=int, default=150)
    parser.add_argument("--solar-max-cycles", type=int, default=1)
    parser.add_argument("--solar-max-speed", type=float, default=float("inf"))
    args = parser.parse_args()
    args.prefix_actions = [int(value) for value in args.prefix_actions.split(",") if value]
    if any(not 0 <= action < 31 for action in args.prefix_actions):
        parser.error("prefix actions must be 0..30")
    if args.policy is None and not 0 <= args.constant_action < 31:
        parser.error("constant action must be 0..30")
    evaluate(args)


if __name__ == "__main__":
    main()
