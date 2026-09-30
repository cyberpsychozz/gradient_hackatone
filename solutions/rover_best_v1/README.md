# Mars Rover: all-wheel drive + pitch control + residual PPO

`train.py` first exports a deterministic ONNX fallback. Its first three decisions are 13, 1, 13: these toggle the transmission from rear to front to all-wheel drive. Thereafter it uses gas (1) and corrects body pitch with gas plus tilt (10 or 11) when `|angle| > 0.1` rad. This policy has no pretrained weights.

The trainer runs recurrent PPO on 64 independent environments by default: 32 standard all-biome worlds, 16 dense-obstacle worlds, and 16 adverse-weather worlds. The stress worlds start at full terrain and biome difficulty, with many more hazards in the first kilometres. PPO uses two epochs, a 128-unit GRU, and the deterministic controller as an action prior. By default its learned part can choose only macros 1, 10 and 11; the first three decisions are scripted as 13, 1, 13. A learned policy replaces the fallback only if it improves on two separate 48-trial mixed-biome sets and retains at least 95% of the fallback median on **both** 48-trial stress sets. The rollout reward removes the engine's repeated cumulative-damage penalty. Training is CPU-only and defaults to 6600 seconds, leaving time for export within the two-hour limit.

The ZIP must contain source files only, without `artifacts/` or checkpoints. Docker builds the bundled native simulator and writes `/output/policy.onnx` before PPO starts.

## Local training

Install the package and PyTorch, ONNX and ONNX Runtime in a Python environment with a C++20 compiler and pybind11, then run:

```bash
python -m pip install --no-build-isolation --no-deps .
python train.py --max-seconds 300 --eval-interval 30
python check_policy.py artifacts/policy.onnx
```

The full CPU run uses the default `--max-seconds 6600`. `--num-envs`, `--rollout-steps`, `--epochs`, `--hidden-size` and `--stress-train-fraction` can be changed for profiling. The stress fraction defaults to 0.5.

To let PPO explore selected additional controls, use `--extra-actions 12,14,16,17,23` (lidar, solar, climb, propeller, ballast). `--extra-actions all` opens every macro after the scripted drive setup. Extra actions start with a -3 logit bias, adjustable with `--extra-logit-bias`; the chosen actions are printed every 50 updates. These options are experimental. The fallback still wins if the learned policy fails the independent evaluation gate. A 75-second selected-macro trial did not beat the fallback; see `docs/rover_experiments.md`.

The evaluation script also accepts `--auto-shift --shift-threshold 10 --prefix-actions 13,1,13 --level-threshold 0.1` to reproduce the rule controller and vary its initial macro sequence. `--solar-rescue-threshold 0.12` tests one stop-and-charge cycle as a separate heuristic.

For an independent local evaluation with an installed `mars_rover_env`:

```bash
python evaluate_onnx.py --policy artifacts/policy.onnx --runs 48 --seed 904
```

The local median is a proxy: the platform uses a separate simulator build and hidden worlds. The supplied evaluation script runs the exported ONNX, holds each action for eight physics frames, and reports maximum distance per 300-second trial. Add `--trace-actions` to count macros before and after each trial's best distance.

## Hard-world evaluation

The source package includes `eval_stress_dense.yaml`, `eval_stress_weather.yaml`
and `eval_stress_ultra.yaml`. The first concentrates craters, steps, biome
transitions and unexpected terrain in the first 3 km. The second also overlays
wind, ice and low gravity and generates cold conditions. The ultra profile
combines both: 130 craters, 60 steps, a Wind/Ice/LowGravity/Mud stack and full
difficulty from the start. They are adversarial local holdouts, not copies of
the organizer's hidden worlds.

From the repository root, after installing the package and ONNX Runtime:

```bash
python solutions/rover_best_v1/stress_bench.py solutions/rover_best_v1/artifacts/policy.onnx --out docs/rover_stress_results.json
```

The command evaluates mixed, dense and weather profiles at seeds 904, 1904 and 2904, with 48 independent trials per cell. It writes every trial distance as well as medians and lower quartiles. To inspect one world visually, pass either stress YAML as the GUI's environment config.
