"""Benchmark a rover ONNX on balanced mixed, dense, and weather holdouts.

The profiles intentionally exceed the public generator's obstacle density.
They are local adversarial tests, not a replica of the platform's hidden worlds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from world_profiles import PROFILES as WORLD_PROFILES


HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "python" / "mars_rover_env" / "configs"
PROFILES = {
    "mixed": None,
    "dense": CONFIGS / "eval_stress_dense.yaml",
    "weather": CONFIGS / "eval_stress_weather.yaml",
    **{profile.name: profile.name for profile in WORLD_PROFILES},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="+", type=Path)
    parser.add_argument("--profiles", default="mixed,dense,weather")
    parser.add_argument("--seeds", default="904,1904,2904")
    parser.add_argument("--runs", type=int, default=48)
    parser.add_argument("--out", type=Path, default=Path("stress_results.json"))
    args = parser.parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not profiles or any(profile not in PROFILES for profile in profiles):
        parser.error(f"--profiles must list choices from {', '.join(PROFILES)}")
    if not seeds or args.runs <= 0:
        parser.error("--seeds must be nonempty and --runs must be positive")
    results = {}
    for policy in args.policies:
        if not policy.is_file():
            parser.error(f"policy does not exist: {policy}")
        policy_results = {}
        for profile in profiles:
            profile_results = {}
            for seed in seeds:
                cmd = [
                    sys.executable, str(HERE / "evaluate_onnx.py"),
                    "--policy", str(policy.resolve()), "--runs", str(args.runs),
                    "--seed", str(seed),
                ]
                config = PROFILES[profile]
                if config is None:
                    cmd.extend(("--biome-split", "0"))
                elif isinstance(config, str):
                    cmd.extend(("--world-profile", config))
                else:
                    cmd.extend(("--config", str(config)))
                completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
                trial = json.loads(completed.stdout)
                profile_results[str(seed)] = {
                    "median_m": trial["median_m"],
                    "q25_m": trial["q25_m"],
                    "q75_m": trial["q75_m"],
                    "mean_m": trial["mean_m"],
                    "distances_m": trial["distances_m"],
                }
                print(f"{policy.name:24s} {profile:8s} seed={seed:<6d} "
                      f"median={trial['median_m']:7.1f} m "
                      f"q25={trial['q25_m']:7.1f} m", flush=True)
            policy_results[profile] = profile_results
        results[str(policy)] = policy_results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
