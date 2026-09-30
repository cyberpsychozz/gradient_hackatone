"""Cross-entropy search over a compact pitch controller on parallel worlds.

Each candidate sees identical terrain seeds in eight independent world
archetypes. The objective is normalized by the original controller's
complete 300 s race distance. Selection uses separate worlds and seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from mixed_env import MixedVecEnv
from model_param_rule import Policy
from train import export_policy
from world_profiles import PROFILES, training_config

TRAIN_WORLDS = (
    "reference", "broken_plain", "ice_crosswind", "soft_swamp",
    "dry_heat", "water_basin", "deep_water", "storm_crust",
)
HOLDOUT_WORLDS = (
    "rolling_hills", "viscous_mud", "shallow_frost", "ice_ridge",
    "sucking_bog", "thermal_marsh", "dense_ridges", "sand_mud_wind",
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}
MACROS = np.asarray(ACTION_MACROS, np.int32)


def make_env(names: tuple[str, ...], population: int) -> MixedVecEnv:
    return MixedVecEnv([
        MarsRoverVecEnv(population, config_override=training_config(PROFILE_BY_NAME[name]))
        for name in names
    ])


def races(env: MixedVecEnv, parameters: np.ndarray, seed: int) -> np.ndarray:
    population = len(parameters)
    groups = len(env.cohorts)
    if env.num_envs != population * groups:
        raise ValueError("one candidate lane required in each world")
    env.reset(seed)
    for group in range(groups):
        course_seed = seed + group * 100_003
        for candidate in range(population):
            env.reset_at(group * population + candidate, course_seed, True)
    thresholds = np.tile(parameters[:, 0], groups)
    gains = np.tile(parameters[:, 1], groups)
    best = env.obs[:, 0].copy() * 1000.0
    done = np.zeros(env.num_envs, bool)
    phase = np.zeros(env.num_envs, np.uint8)
    for _ in range(2250):
        tilt = env.obs[:, 4] + gains * env.obs[:, 5] * 10.0
        actions = np.where(tilt > thresholds, 11,
                           np.where(tilt < -thresholds, 10, 1)).astype(np.intp)
        actions[phase == 0] = 13
        actions[phase == 1] = 1
        actions[phase == 2] = 13
        np.minimum(phase + 1, 3, out=phase)
        controls = MACROS[actions].copy()
        controls[done] = 0
        for _ in range(8):
            if done.all():
                break
            _, _, terminated, truncated, _ = env.step_uint8(controls)
            active = ~done
            np.maximum(best, env.obs[:, 0] * 1000.0, out=best, where=active)
            done |= terminated.astype(bool) | truncated.astype(bool)
            controls[done] = 0
        if done.all():
            break
    return np.maximum(best.reshape(groups, population) - 1.0, 0.0)


def score(distances: np.ndarray, robust_penalty: float = 0.0) -> np.ndarray:
    baseline = distances[:, 0:1]
    # Relative distance makes a severe 200 m world matter alongside an easy
    # 1000 m world, while clipping limits the influence of one lucky jump.
    relative = (distances - baseline) / np.maximum(baseline, 100.0)
    relative = np.clip(relative, -1.0, 1.0)
    return np.mean(relative, axis=0) - robust_penalty * np.std(relative, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--training-repeats", type=int, default=1,
                        help="Independent course seeds per world and candidate.")
    parser.add_argument("--robust-penalty", type=float, default=0.0,
                        help="Penalty on variation of relative gains between worlds.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "artifacts/cem_rule/policy.onnx")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).parent / "artifacts/cem_rule/results.json")
    args = parser.parse_args()
    if args.population < 5 or args.generations < 1 or args.training_repeats < 1:
        parser.error("population >= 5, generations >= 1, repeats >= 1 required")
    if args.robust_penalty < 0:
        parser.error("robust penalty must be nonnegative")
    rng = np.random.default_rng(args.seed)
    train_env = make_env(TRAIN_WORLDS * args.training_repeats, args.population)
    mean = np.array([0.1, 0.0])
    std = np.array([0.08, 0.15])
    best_parameters = mean.copy()
    best_score = 0.0
    rows = []
    start = time.monotonic()
    for generation in range(args.generations):
        candidates = rng.normal(mean, std, (args.population, 2))
        candidates[:, 0] = np.clip(candidates[:, 0], 0.02, 0.45)
        candidates[:, 1] = np.clip(candidates[:, 1], -0.5, 0.6)
        candidates[0] = [0.1, 0.0]
        candidates[1] = best_parameters
        distances = races(train_env, candidates, args.seed + 1_000)
        scores = score(distances, args.robust_penalty)
        order = np.argsort(scores)[::-1]
        elite = candidates[order[:max(3, args.population // 5)]]
        mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
        std = np.maximum([0.015, 0.025], 0.6 * std + 0.4 * elite.std(axis=0))
        winner = int(order[0])
        if scores[winner] > best_score:
            best_score = float(scores[winner])
            best_parameters = candidates[winner].copy()
        row = {"generation": generation + 1, "best_relative": round(float(scores[winner]), 4),
               "alltime_relative": round(best_score, 4),
               "threshold": round(float(candidates[winner, 0]), 4),
               "angular_gain": round(float(candidates[winner, 1]), 4),
               "elapsed_s": round(time.monotonic() - start, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Independent worlds and seed. Candidate zero is the exact fallback.
    holdout_env = make_env(HOLDOUT_WORLDS, 2)
    holdout = races(holdout_env,
                    np.stack(([0.1, 0.0], best_parameters)),
                    args.seed + 900_000)
    gains = holdout[:, 1] - holdout[:, 0]
    report = {
        "algorithm": "CEM over pitch threshold and angular velocity gain",
        "training_worlds": TRAIN_WORLDS, "holdout_worlds": HOLDOUT_WORLDS,
        "seed": args.seed, "population": args.population,
        "generations": args.generations,
        "training_repeats": args.training_repeats,
        "robust_penalty": args.robust_penalty, "history": rows,
        "best_parameters": best_parameters.tolist(),
        "holdout_base_m": holdout[:, 0].round(2).tolist(),
        "holdout_candidate_m": holdout[:, 1].round(2).tolist(),
        "holdout_mean_difference_m": float(np.mean(gains)),
        "holdout_median_difference_m": float(np.median(gains)),
        "holdout_world_wins": int(np.sum(gains > 0.5)),
        "holdout_world_losses": int(np.sum(gains < -0.5)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    export_policy(Policy(*best_parameters), args.output)
    print("HOLDOUT " + json.dumps({k: report[k] for k in (
        "best_parameters", "holdout_mean_difference_m",
        "holdout_median_difference_m", "holdout_world_wins", "holdout_world_losses",
    )}), flush=True)


if __name__ == "__main__":
    main()
