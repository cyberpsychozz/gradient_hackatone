# Rover experiments and interpretation

## What the supplied `message.txt` shows

One `dinasty` update reports 65,536 physics frames (`128 environments × 64 decisions × 8 physics steps`), equal to only 512 physics frames or 8.53 simulated seconds per environment. At roughly 87,000 aggregate physics frames/s, 154 million frames in 1,770 s is plausible for the compiled, vectorized C++ simulator. It does not mean the policy has converged.

The `median_distance` printed for each update often covers only 0–5 trials; `nan` means no trial finished in that update. It should not be used for model selection. In the supplied excerpt, independent deterministic eval rose from 211.5 to 250.6 m on the regular configuration while hard eval rose from 97.7 to 172.7 m. Later hard eval reached 202.6 m while regular eval fell to 229.4 m. The original `quick_eval` also divided the 18,000-frame trial budget by eight and then counted physics frames, causing `trial_progress` to reach 1 eight times too soon. The revised evaluator fixes this and feeds the sum of eight rewards.

The training reward in the native engine subtracts `hard_contact_penalty × state.damage` on **every** frame, although damage is cumulative. `stuck` is always false in the current step path. The new trainer sets the cumulative-damage coefficient to zero and keeps progress, energy, and flip terms. The evaluator score remains the maximum reached distance, so model selection uses that metric rather than the training reward.

## Local full-trial results

Every number below is a median of **48 independent 300-second trials** from the exported ONNX or the matching scripted prototype. They are local proxies, not official hidden scores. The regular configuration uses the current local native environment with 40 biomes (20 original and 20 added training) in its catalog and `biome_split=1`; the mixed configuration sets `biome_split=0` to sample all 40. The hard configuration uses the `dinasty` native build and a larger difficulty offset.

| Controller | Regular seed 904 | Regular seed 1904 | Regular seed 2904 | Hard seed 904 | Hard seed 1904 | All 40 seed 904 | All 40 seed 1904 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always gas (action 1) | 199.5 | — | — | — | — | 178.4 | — |
| Front drive then gas | 381.7 | 394.0 | 388.5 | 243.0 | 204.9 | — | — |
| Front drive + pitch correction, 0.2 rad | 448.9 | 408.2 | 474.0 | 326.4 | 221.3 | 275.6 | 272.6 |
| All-wheel drive + pitch correction, 0.2 rad | 507.4 | 522.2 | 451.6 | 337.9 | 257.0 | 397.0 | 370.0 |
| **All-wheel drive + pitch correction, 0.1 rad (selected)** | **539.3** | **521.3** | **444.4** | **332.3** | **307.4** | **427.4** | **361.9** |

The all-wheel policy starts with actions `13, 1, 13` because the drive-mode switch triggers only on a fresh press. It then chooses gas (1), gas plus left tilt (10), or gas plus right tilt (11) from body angle. A 0.1-radian threshold improved the aggregate median across the mixed and hard suites; normal-suite results were similar to 0.2. A further seed-3904 check gave mixed/hard medians of 387/334 m at 0.1 versus 351/312 m at 0.2. We therefore selected 0.1, while keeping both results above for transparency.

A separate check on eight fixed public anchor biomes (six seeds each) found a higher median than constant gas in every one. The weakest absolute medians were mud (188.5 m) and liquid (233.2 m), so unfamiliar terrain with sustained drag remains a risk.

The ONNX implementation of the all-wheel controller reproduced the scripted per-trial distances exactly on the regular, hard, and all-40 seed-904 suites. It has seven inputs, two outputs, 63 nodes, no operators outside the platform allowlist, and `rover.format=rover-policy-onnx-v1`. The source ZIP excludes this generated ONNX and all checkpoints.

## Learning attempts

- Unrestricted recurrent PPO, 25.6 million frames in 300 CPU seconds: no fixed-seed improvement over its front-drive fallback; eval collapsed after early updates.
- Three-action residual PPO over front drive plus pitch correction, 24.4 million frames in 300 CPU seconds: reached 478.8 m on one fixed 16-trial eval, but on independent 48-trial suites it scored 447.3 m regular and 301.1 m hard versus 448.9 and 326.4 m for the deterministic controller. The fixed 16-trial improvement did not generalize.
- Three-action residual PPO over all-wheel drive, 24.1 million frames in 300 CPU seconds: its best fixed 16-trial eval improved from 276.9 to 359.5 m. On independent 48-trial suites the learned checkpoint improved the three regular medians from 507/522/452 to 537/531/488 m, but decreased both hard medians from 338/257 to 316/228 m and one all-40 median from 370 to 337 m. The gain is not yet robust enough to replace the deterministic fallback.

`train.py` exports the fallback before training and replaces it only after improving two independent 48-trial evaluations while maintaining at least 95% of its hard-terrain median. The default CPU budget is 6,600 seconds.

For local reproduction, build `solutions/rover_best_v1` and run `python evaluate_onnx.py --policy artifacts/policy.onnx --runs 48 --seed 904`. Use `--biome-split 0` for all 40 local biomes, or `--config python/mars_rover_env/configs/eval_hard.yaml` for a hard configuration. The platform uses separate hidden worlds, so no local score is guaranteed to transfer.

The experimental design follows the [original PPO paper](https://arxiv.org/abs/1707.06347) for clipped policy updates and the [Stable-Baselines3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) on separate evaluation, repeated runs, and reward engineering. Reward shaping is kept modest because arbitrary extra rewards can change the optimal policy; see the [potential-based shaping result](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

## Why the other macros are absent from the selected policy

The default residual policy deliberately masks 28 of the 31 macros with a -10000 logit. After the scripted 13, 1, 13 drive-mode setup, only 1 (gas), 10 (gas + left pitch), and 11 (gas + right pitch) are selectable. The exported weight-free fallback reads body angle for pitch control and does not use the other 159 observation channels for decisions. The non-use of lidar, solar charging, ballast, and other controls is therefore a design choice, **not evidence that PPO tried them and learned that they are useless**.

The mechanics make some macros strongly contextual. Lidar makes distant terrain samples visible for a short time but spends energy, so it cannot help an angle-only controller. Solar charging toggles a persistent mode that sets throttle to zero and brakes; it charges after the panel deploys and the rover is stationary or airborne. Ballast changes buoyancy in liquid zones, while sustained propeller use spends energy even on dry terrain. Climb mode improves traction in some conditions but adds drag. One-shot or always-on action comparisons therefore provide only a first screen, not a full test of situational use.

Matched 48-trial fixed-biome checks (seed 904, no automatic shifting) gave:

| Experiment | Baseline median | Modified median |
| --- | ---: | ---: |
| Liquid: propeller toggled on after drive setup | 218 m | 40 m |
| Liquid: ballast blown for 14 decisions after drive setup | 218 m | 18 m |
| Mud: climb mode toggled on after drive setup | 221 m | 209 m |
| Mud: one solar charge cycle once energy is below 12% | 221 m | 238 m |

The mud solar result reproduced more weakly on seed 1904 (137 to 142 m). It did not improve the mixed all-40 medians, and the unqualified charge heuristic decreased two matched hard-suite medians (355 to 347 m and 324 to 321 m). Restricting charging to speeds below 0.6 m/s preserved those hard medians but reduced one mixed median (362 to 354 m). It was not added to the submitted fallback.

The saved historical `dinasty` ONNX exposed 30 effective macros: its training source masked macro 18 at the time because that bit has no decoded control. The current `dinasty` and `ppo_recurrent_v1` sources expose all 31 macros. On 16 matched seed-904 trials, the saved `dinasty` ONNX selected solar macro 14 **2,491 times**, brake 2 **3,089 times**, and other controls as well. An action-timing trace found **zero solar decisions before each run's final maximum distance**; all 2,491 happened afterward. It selected neither lidar 12 nor ballast 23 in these trials. Every per-run maximum distance matched an always-gas controller on the same seeds. Thus this model can issue special actions but has not learned to use them to extend the scored distance. Reproduce the timing with `solutions/evaluate_onnx.py --policy solutions/dinasty/artifacts/policy.onnx --runs 16 --seed 904 --trace-actions` using the matching installed `dinasty` environment.

The trainer now has an explicit `--extra-actions` switch and logs sampled macro counts. A 12-second run with actions 12, 14, 16, 17, and 23 enabled sampled all five. In a 75-second run (2.42 million frames), the selected-action PPO worsened its 16-trial evaluation median from 628 to 471 and then 276 m; the hard median fell from 331 to 230 m. A separate 45-second all-31 run also lost ground on its small evaluation suite. The fallback export gate retained the angle controller. These short trials do not rule out useful timed interventions; they show why indiscriminately opening the action space is not yet a verified improvement.

## Dense obstacles and adverse weather holdouts

The local default generates 68 craters and 30 steps along a 20 km sampled terrain. At seed 904 its starting terrain difficulty reports 0.285. This sparsity lets simple controls travel several hundred metres without meeting many generated obstacles. The old `difficulty_distance_offset` changes terrain difficulty but does not raise the separate biome-chain difficulty. We found that the opt-in `force_full_difficulty` flag had been parsed but never used by that biome difficulty calculation. The native simulator now applies it, and the holdout figures below were rerun after this fix.

The dense profile samples 3 km with 110 craters, 52 steps, 20 short biome zones, frequent terrain surprises, and all 40 local biomes. The weather profile samples 3 km with 96 craters and 45 steps, plus a persistent Wind/Ice/LowGravity mechanism stack, producing cold, slippery, gusty and low-gravity conditions. Both set terrain and biome difficulty to maximum from the start, keep the 5 m safe spawn surface, and retain the 300-second action/score protocol. They are deliberately harder local proxies; the hidden platform generator is not available.

The current deterministic ONNX gave the following medians (metres) on independent 48-trial, 300-second sets:

| Local profile | Seed 904 | Seed 1904 | Seed 2904 |
| --- | ---: | ---: | ---: |
| Mixed local biomes | 427.4 | 361.9 | 311.5 |
| Dense obstacles | 64.2 | 72.9 | 56.9 |
| Weather stack | 57.1 | 58.2 | 56.7 |

The corrected stress suites expose a clear generalization weakness: their medians are about 57–73 m, versus 312–427 m on mixed worlds. The hidden platform generator is unavailable, so these are local proxies rather than predicted platform scores.

The two YAML stress profiles are now held out from PPO training and remain separate 48-trial model-selection gates. A 20-second integration run of the new 12-profile mixture completed 473,088 frames on 16 parallel environments and correctly kept the fallback when PPO did not improve. Run `solutions/rover_best_v1/stress_bench.py` to regenerate `docs/rover_stress_results.json`; the JSON contains all trial distances.

## Diverse training worlds

The earlier training batch had only three static profile types: regular, dense and weather. The new batch has 12 simultaneous archetypes, each with changing terrain and biome seeds. At 64 environments, 32 are accessible, 24 are challenging and 8 are severe. The severe 12.5% receives dense obstacle fields or adverse stacked mechanics at full biome difficulty and elevated terrain difficulty; the other 87.5% preserves enough traversable distance for long-horizon progress learning. The two adversarial evaluation YAMLs are not used as training cohorts.

The archetypes vary physical course length (3.5–6 km where explicitly set), craters (44–96), steps (18–43), biome transition length, terrain surprise rate, terrain and biome difficulty ramps, starting temperature/energy, and persistent sand/mud/ice/wind/low-gravity/crust layers. All 40 local biomes are eligible, and each reset samples new positions and mechanic parameters. The reference archetype retains the ordinary world configuration.

A reproducible 24-trial local calibration of the existing deterministic controller gave accessible medians of 318–907 m, challenging medians of 219–541 m, and severe medians of 115–154 m. The deliberately adjusted sand/mud/wind profile scored 304 m; its former sand/wind version scored 846 m and was too easy for its intended tier. See `docs/rover_world_profiles_results.json` for every profile and `solutions/rover_best_v1/world_profiles_bench.py` to rerun it. The short 16-environment PPO smoke test reached 473,088 simulator frames in 20 s; it verifies throughput and integration, not a trained policy improvement.

## Full CPU training after world diversification

Five local PPO runs on the 12-profile world mixture processed approximately 384 million physics frames. Only the reward-scaled variant passed the old two-mixed/two-stress selection gate, and its improvement was only 2.4 m on the weaker mixed median. A further 864 matched trials across six unseen seed sets showed a mean difference of -0.35 m and no robust gain. The deterministic ONNX remains the selected artifact. Full settings, logs, and paired comparisons are recorded in `docs/rover_training_runs.md`.

## 24-world follow-up

The 12-profile findings above are historical. Three later 105-minute PPO runs used 24 world profiles and 64 parallel simulations. The selected separate-critic PPO with seven extra macros gained 33.0 m mean distance over the fallback on 768 unseen paired local races (95% bootstrap interval 25.5–41.2 m); it also gained 21.2 m on 576 full-difficulty dense/weather races. The wide-action separate-critic variant is retained as a second candidate. The source-only packages and Russian summary are in `best_solutions/` and `Best_solutions.md`, with settings and per-race data in `docs/rover_24_world_training.md`. Neither package has a hidden-platform score yet.
