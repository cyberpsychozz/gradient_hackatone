# Mars Rover: all-wheel drive, recurrent PPO, and CEM

`train.py` first exports a deterministic ONNX fallback. Its first three decisions are 13, 1, 13: these toggle the transmission from rear to front to all-wheel drive. Thereafter it uses gas (1) and corrects body pitch with gas plus tilt (10 or 11) when `|angle| > 0.1` rad. This policy has no pretrained weights.

The trainer runs recurrent PPO on 64 independent environments by default: 32 accessible, 24 challenging, and 8 severe worlds across 24 terrain/weather profiles. Each profile generates new terrain, weather parameters, and obstacle positions from its own seed. Some profiles fix a named biome and its physical layer to guarantee swamp, ice, heat, or water; the rest vary across the full biome bank. The severe share is 12.5% of the default batch, so early failures do not dominate PPO. PPO uses two epochs, a 128-unit GRU, and the deterministic controller as an action prior. By default its learned part can choose only macros 1, 10 and 11; the first three decisions are scripted as 13, 1, 13. A learned policy replaces the fallback only if its weaker median across two separate 48-trial mixed-biome sets improves by at least 10 m and it retains at least 95% of the fallback median on **both** 48-trial stress sets. The rollout reward removes the engine's repeated cumulative-damage penalty. Training is CPU-only and defaults to 6600 seconds, leaving time for export within the two-hour limit.

The ZIP must contain source files only, without `artifacts/` or checkpoints. Docker builds the bundled native simulator and writes `/output/policy.onnx` before PPO starts.

## Local training

Install the package and PyTorch, ONNX and ONNX Runtime in a Python environment with a C++20 compiler and pybind11, then run:

```bash
python -m pip install --no-build-isolation --no-deps .
python train.py --algorithm separate_ppo --seed 2029 \
  --extra-actions 12,14,16,17,23,24,28 --extra-logit-bias -4 \
  --max-seconds 6300 --eval-interval 1200 \
  --output artifacts/local_context/policy.onnx \
  --checkpoint artifacts/local_context/latest.pt
python check_policy.py artifacts/local_context/policy.onnx
```

The full CPU run uses the default `--max-seconds 6600` and evaluates every 20 minutes. PPO defaults to a 0.2 return scale, 0.02 rule deviation penalty, and a learning rate decreasing from 0.00018 to 0.00007; these settings reduced value loss in a 30-minute local run. `--num-envs`, `--rollout-steps`, `--epochs`, `--hidden-size`, `--stress-train-fraction`, `--severe-train-fraction`, `--return-scale`, `--rule-coef`, and `--min-improvement-m` can be changed for profiling. The stress share defaults to 0.5 and includes the severe share; the severe share defaults to 0.12. The minimum mixed-suite improvement defaults to 10 m. The 24 archetypes and their calibrated 24-run distances are in `world_profiles.py` and `docs/rover_world_profiles_24_results.json`.

To use a feedforward critic independent of the recurrent actor, pass `--algorithm separate_ppo`. To let PPO explore selected additional controls, use `--extra-actions 12,14,16,17,23` (lidar, solar, climb, propeller, ballast). `--extra-actions all` opens every macro after the scripted drive setup. Extra actions start with a -3 logit bias, adjustable with `--extra-logit-bias`; the chosen actions are printed every 50 updates. These options are experimental. The fallback still wins if the learned policy fails the independent evaluation gate. A 75-second selected-macro trial did not beat the fallback; see `docs/rover_experiments.md`.

For direct evolutionary optimization of the pitch controller on eight parallel world types, run `python cem_rule.py --population 24 --generations 48`; its separate world and seed holdout is reported in `artifacts/cem_rule/results.json`. A second CEM run used `--training-repeats 2 --robust-penalty 0.4`, but neither candidate improved independent holdouts reliably.

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

## Training world mixture

At the default 64 environments, the 24 simultaneous archetypes allocate 32 accessible, 24 challenging and 8 severe environments. They vary obstacle density, terrain frequency, biome transition length, surprise terrain, starting energy and temperature, and mechanism overlays (sand, mud, ice, wind, low gravity, crust and liquid). Named swamp, ice, heat and water profiles fix both the biome and physical layer. All 40 local biomes (20 original and 20 added training biomes) remain available on every nonfixed profile. Each reset draws a fresh world seed. `world_profiles_bench.py --runs 24 --out profile_results.json` measures the current fallback on each profile; the results are local calibration, not platform scores. The adversarial `eval_stress_dense.yaml` and `eval_stress_weather.yaml` are kept out of the training mixture for independent model selection.

The training mixture changes which experiences PPO sees. It does not expand the default policy's three trainable drive/tilt macros. Use `--extra-actions` for experiments with solar, lidar, ballast or other controls, and keep the independent evaluation gate enabled.

## Full 24-world experiments

Three 105-minute CPU runs processed 240–266 million physics frames each. On 768 paired unseen trials across eight profiles, the selected separate-critic PPO with restricted macros gained 33.0 m mean distance over the fallback (95% paired bootstrap interval 25.5–41.2 m). On another 576 full-difficulty dense/weather trials it gained 21.2 m mean (interval 16.1–26.6 m). The wide-action separate-critic PPO gained 18.8 m mean on that full-difficulty set, but was weaker on the mixed and aquatic worlds. These are **local** results; the new source ZIPs have not been tested on the platform. The selected packages and local preview ONNX files are in `best_solutions/`; methods, metrics and limitations are in `Best_solutions.md` and `docs/rover_24_world_training.md`.

## Earlier 12-world training result

Five CPU runs processed about 384 million simulator frames in total. The best learned checkpoint passed the earlier local selection gate by just 2.4 m. On 864 additional paired trials across mixed, dense, and weather worlds, its mean distance differed from the fallback by -0.35 m (95% bootstrap interval -2.88 to +2.17 m). The original local artifact in `artifacts/policy.onnx` therefore remains the deterministic fallback; the experimental learned model is preserved at `artifacts/scaled/policy.onnx`. See `docs/rover_training_runs.md` for the settings, outcomes, and independent results. These local trials do not determine the hidden platform score.
