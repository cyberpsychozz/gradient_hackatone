"""Summarize paired ONNX race results from stress_bench.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path, bootstrap: int = 3000) -> dict:
    data = json.loads(path.read_text())
    names = list(data)
    if len(names) < 2:
        raise ValueError("benchmark must contain a baseline and a candidate")
    baseline = data[names[0]]
    rng = np.random.default_rng(120039)
    output = {"baseline": names[0], "candidates": {}}
    for candidate in names[1:]:
        cells = []
        by_profile = {}
        for profile, seeds in baseline.items():
            profile_differences = []
            for seed, base in seeds.items():
                other = data[candidate][profile][seed]
                left = np.asarray(base["distances_m"], dtype=np.float64)
                right = np.asarray(other["distances_m"], dtype=np.float64)
                if left.shape != right.shape:
                    raise ValueError(f"unpaired result for {profile}/{seed}")
                differences = right - left
                cells.append(differences)
                profile_differences.extend(differences.tolist())
            by_profile[profile] = {
                "runs": len(profile_differences),
                "mean_difference_m": round(float(np.mean(profile_differences)), 2),
                "median_difference_m": round(float(np.median(profile_differences)), 2),
                "wins": int(np.sum(np.asarray(profile_differences) > 0.5)),
                "losses": int(np.sum(np.asarray(profile_differences) < -0.5)),
            }
        all_differences = np.concatenate(cells)
        sampled = np.empty(bootstrap)
        for rep in range(bootstrap):
            sampled[rep] = np.mean([np.mean(cell[rng.integers(0, len(cell), len(cell))])
                                    for cell in cells])
        output["candidates"][candidate] = {
            "runs": len(all_differences),
            "mean_difference_m": round(float(np.mean(all_differences)), 2),
            "median_difference_m": round(float(np.median(all_differences)), 2),
            "mean_95pct_ci_m": np.quantile(sampled, [0.025, 0.975]).round(2).tolist(),
            "wins": int(np.sum(all_differences > 0.5)),
            "losses": int(np.sum(all_differences < -0.5)),
            "by_profile": by_profile,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(args.results)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
