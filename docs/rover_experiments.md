# Rover experiments and interpretation

## What the supplied `message.txt` shows

One `dinasty` update reports 65,536 physics frames (`128 environments × 64 decisions × 8 physics steps`), equal to only 512 physics frames or 8.53 simulated seconds per environment. At roughly 87,000 aggregate physics frames/s, 154 million frames in 1,770 s is plausible for the compiled, vectorized C++ simulator. It does not mean the policy has converged.

The `median_distance` printed for each update often covers only 0–5 trials; `nan` means no trial finished in that update. It should not be used for model selection. In the supplied excerpt, independent deterministic eval rose from 211.5 to 250.6 m on the regular configuration while hard eval rose from 97.7 to 172.7 m. Later hard eval reached 202.6 m while regular eval fell to 229.4 m. The original `quick_eval` also divided the 18,000-frame trial budget by eight and then counted physics frames, causing `trial_progress` to reach 1 eight times too soon. The revised evaluator fixes this and feeds the sum of eight rewards.

The training reward in the native engine subtracts `hard_contact_penalty × state.damage` on **every** frame, although damage is cumulative. `stuck` is always false in the current step path. The new trainer sets the cumulative-damage coefficient to zero and keeps progress, energy, and flip terms. The evaluator score remains the maximum reached distance, so model selection uses that metric rather than the training reward.

## Local full-trial results

Every number below is a median of **48 independent 300-second trials** from the exported ONNX or the matching scripted prototype. They are local proxies, not official hidden scores. The regular configuration uses the current public native environment with 40 biomes in its catalog and `biome_split=1`; the mixed configuration sets `biome_split=0` to sample all 40. The hard configuration uses the `dinasty` native build and a larger difficulty offset.

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

For local reproduction, build `solutions/rover_best_v1` and run `python evaluate_onnx.py --policy artifacts/policy.onnx --runs 48 --seed 904`. Use `--biome-split 0` for all public biomes, or `--config python/mars_rover_env/configs/eval_hard.yaml` for a hard configuration. The platform uses separate hidden worlds, so no local score is guaranteed to transfer.

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

The earlier `dinasty` and `ppo_recurrent_v1` models differ: their action head exposes 30 effective macros (only macro 18 is masked because its bit has no decoded control). On 16 matched seed-904 trials, the saved `dinasty` ONNX selected solar macro 14 **2,491 times**, brake 2 **3,089 times**, and other controls as well. An action-timing trace found **zero solar decisions before each run's final maximum distance**; all 2,491 happened afterward. It selected neither lidar 12 nor ballast 23 in these trials. Every per-run maximum distance matched an always-gas controller on the same seeds. Thus this model can issue special actions but has not learned to use them to extend the scored distance. Reproduce the timing with `solutions/evaluate_onnx.py --policy solutions/dinasty/artifacts/policy.onnx --runs 16 --seed 904 --trace-actions` using the matching installed `dinasty` environment.

The trainer now has an explicit `--extra-actions` switch and logs sampled macro counts. A 12-second run with actions 12, 14, 16, 17, and 23 enabled sampled all five. In a 75-second run (2.42 million frames), the selected-action PPO worsened its 16-trial evaluation median from 628 to 471 and then 276 m; the hard median fell from 331 to 230 m. A separate 45-second all-31 run also lost ground on its small evaluation suite. The fallback export gate retained the angle controller. These short trials do not rule out useful timed interventions; they show why indiscriminately opening the action space is not yet a verified improvement.

## Dense obstacles and adverse weather holdouts

The public default generates 68 craters and 30 steps along a 20 km sampled terrain. At seed 904 its starting terrain difficulty reports 0.285. This sparsity lets simple controls travel several hundred metres without meeting many generated obstacles. The old `difficulty_distance_offset` changes terrain difficulty but does not raise the separate biome-chain difficulty. We added an opt-in `force_full_difficulty` generator flag for local stress worlds, while keeping ordinary worlds unchanged.

The dense profile samples 3 km with 110 craters, 52 steps, 20 short biome zones, frequent terrain surprises, and all 40 public biomes. The weather profile samples 3 km with 96 craters and 45 steps, plus a persistent Wind/Ice/LowGravity mechanism stack, producing cold, slippery, gusty and low-gravity conditions. Both set terrain and biome difficulty to maximum from the start, keep the 5 m safe spawn surface, and retain the 300-second action/score protocol. They are deliberately harder local proxies; the hidden platform generator is not available.

The current deterministic ONNX gave the following medians (metres) on independent 48-trial, 300-second sets:

| Local profile | Seed 904 | Seed 1904 | Seed 2904 |
| --- | ---: | ---: | ---: |
| Mixed public biomes | 427.4 | 361.9 | 311.5 |
| Dense obstacles | 122.1 | 98.4 | 68.5 |
| Weather stack | 90.1 | 77.2 | 90.7 |

A constant-gas controller scored only 32.5 m on dense and 34.5 m on weather (24 seed-904 trials each). The stress suites therefore expose a genuine generalization weakness without making every trial fail at the spawn. A small control sweep found front-wheel drive promising on one dense seed but substantially worse under weather; no unverified change was made to the exported policy.

The default trainer now allocates half of its 64 parallel simulators to the two stress profiles and uses them as separate 48-trial model-selection gates. A 35-second integration run completed 827,392 frames, evaluated both stress profiles, and correctly kept the fallback when PPO worsened their medians. Run `solutions/rover_best_v1/stress_bench.py` to regenerate `docs/rover_stress_results.json`; the JSON contains all trial distances.
