"""Train a recurrent PPO rover policy with the public vector simulator.

SOLUTION NAME: dinastiya

Training mixes standard worlds with dense-obstacle and adverse-weather
cohorts at full difficulty from the start. Biome chains and layered
mechanics are sampled over the full 40-biome pool, biome callbacks are
dispatched for every biome (sand sinking, mud viscosity, phase worlds), and
the model receives the 160-observation contract. Reward shaping below is
applied to the training copy only; the exported policy is evaluated on the
organizer build.

Robustness: the run uses the full wall-clock budget, exports only finite
weights, atomically replaces /output/policy.onnx, and keeps the checkpoint
with the best deterministic median distance found so far. A divergence of
the final weights therefore cannot erase a good model.
"""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import load_env_config

SOLUTION_NAME = "rover_best_v1"
from mixed_env import MixedVecEnv
from model_residual import Policy
from rule_policy import RulePolicy


def is_finite(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            return False
    for buffer in model.buffers():
        if not torch.isfinite(buffer).all():
            return False
    return True


def export_policy(model: Policy, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_finite(model):
        print("WARNING: skipping export, model weights are not finite", flush=True)
        return False
    module = copy.deepcopy(model).cpu().eval()
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        try:
            from arena.protocol import export_onnx
        except ModuleNotFoundError as exc:
            if exc.name != "arena":
                raise
            # The local environment lacks arena-base. The server uses its validator.
            import onnx

            n = 2
            example = (
                torch.zeros(n, module.obs_dim), torch.zeros(n, dtype=torch.int64),
                torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.ones(n),
                torch.zeros(n, module.hidden_size),
            )
            inputs = (
                "observation", "previous_action", "previous_reward",
                "previous_done", "trial_progress", "trial_start", "memory",
            )
            torch.onnx.export(
                module, example, str(tmp_path), opset_version=17, dynamo=False,
                input_names=list(inputs), output_names=["logits", "next_memory"],
                dynamic_axes={name: {0: "batch"} for name in (*inputs, "logits", "next_memory")},
            )
            graph = onnx.load(str(tmp_path))
            entry = graph.metadata_props.add()
            entry.key, entry.value = "rover.format", "rover-policy-onnx-v1"
            entry = graph.metadata_props.add()
            entry.key, entry.value = "rover.solution", SOLUTION_NAME
            onnx.checker.check_model(graph)
            onnx.save(graph, str(tmp_path))
        else:
            export_onnx(str(tmp_path), module, obs_dim=module.obs_dim,
                        memory_size=module.hidden_size)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"exported {path} ({path.stat().st_size:,} bytes)", flush=True)
    return True


def collect(env, policy, state, cfg, device, rng):
    keys = (
        "obs", "prev_action", "prev_reward", "prev_done", "progress",
        "trial_start", "action", "logprob", "value", "reward", "done",
    )
    data = {key: [] for key in keys}
    start_memory = state["memory"].detach().clone()
    macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    finished_distances = []
    budget = max(1, round(env.trial_time_limit * 60))
    policy.eval()

    for _ in range(cfg.rollout_steps):
        obs = torch.from_numpy(state["obs"].copy()).to(device)
        features = (
            obs, state["prev_action"], state["prev_reward"],
            state["prev_done"], state["progress"], state["trial_start"],
        )
        with torch.no_grad():
            logits, value, next_memory = policy.step(*features, state["memory"])
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            logprob = distribution.log_prob(action)
        for key, tensor in zip(keys, (*features, action, logprob, value)):
            data[key].append(tensor)

        controls = macro[action.cpu().numpy()].copy()
        rewards = np.zeros(cfg.num_envs, dtype=np.float32)
        done = np.zeros(cfg.num_envs, dtype=bool)
        frame_counts = np.zeros(cfg.num_envs, dtype=np.int32)
        for _ in range(cfg.frame_skip):
            _, step_reward, terminated, truncated, _ = env.step_uint8(controls)
            active = ~done
            rewards[active] += step_reward[active]
            frame_counts[active] += 1
            state["best"][active] = np.maximum(
                state["best"][active], env.obs[active, 0] * 1000.0
            )
            done |= terminated.astype(bool) | truncated.astype(bool)
            controls[done] = 0
            if done.all():
                break

        state["steps"] += frame_counts
        trial_start = np.zeros(cfg.num_envs, dtype=np.float32)
        for i in np.flatnonzero(done):
            finished_distances.append(max(0.0, float(state["best"][i] - 1.0)))
            new_trial = bool(env.next_trial_start[i])
            env.reset_at(int(i), seed=int(rng.integers(0, 2**31)),
                         trial_start=new_trial)
            if new_trial:
                state["steps"][i] = 0
                trial_start[i] = 1.0
            state["best"][i] = env.obs[i, 0] * 1000.0

        data["reward"].append(torch.from_numpy(rewards).to(device))
        data["done"].append(torch.from_numpy(done.astype(np.float32)).to(device))
        state["obs"] = env.obs.copy()
        state["prev_action"] = action
        state["prev_reward"] = torch.from_numpy(rewards).to(device)
        state["prev_done"] = torch.from_numpy(done.astype(np.float32)).to(device)
        state["progress"] = torch.from_numpy(
            np.minimum(state["steps"] / budget, 1.0).astype(np.float32)
        ).to(device)
        state["trial_start"] = torch.from_numpy(trial_start).to(device)
        state["memory"] = next_memory

    with torch.no_grad():
        _, bootstrap, _ = policy.step(
            torch.from_numpy(state["obs"]).to(device), state["prev_action"],
            state["prev_reward"], state["prev_done"], state["progress"],
            state["trial_start"], state["memory"],
        )
    batch = {key: torch.stack(value) for key, value in data.items()}
    batch["start_memory"] = start_memory
    return batch, bootstrap, finished_distances


def update_policy(policy, optimizer, batch, bootstrap, cfg, rng):
    reward, done, value = batch["reward"], batch["done"], batch["value"]
    advantage = torch.empty_like(reward)
    gae = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(cfg.rollout_steps)):
        alive = 1.0 - done[t]
        delta = reward[t] + cfg.gamma * next_value * alive - value[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * alive * gae
        advantage[t] = gae
        next_value = value[t]
    returns = advantage + value
    advantage = (advantage - advantage.mean()) / (
        advantage.std(unbiased=False) + 1e-8
    )
    policy.train()
    metrics = []
    chunks = max(1, (cfg.num_envs + cfg.envs_per_batch - 1) // cfg.envs_per_batch)
    for _ in range(cfg.epochs):
        for indices in np.array_split(rng.permutation(cfg.num_envs), chunks):
            ids = indices.tolist()
            logits, predicted_value = policy.sequence(
                *(batch[key][:, ids] for key in (
                    "obs", "prev_action", "prev_reward", "prev_done",
                    "progress", "trial_start",
                )),
                batch["start_memory"][ids],
            )
            distribution = Categorical(logits=logits)
            logprob = distribution.log_prob(batch["action"][:, ids])
            ratio = (logprob - batch["logprob"][:, ids]).exp()
            selected_adv = advantage[:, ids]
            actor_loss = -torch.minimum(
                ratio * selected_adv,
                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * selected_adv,
            ).mean()
            critic_loss = 0.5 * (
                predicted_value - returns[:, ids]
            ).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            metrics.append((actor_loss.item(), critic_loss.item(), entropy.item()))
    return np.mean(metrics, axis=0)


def quick_eval(policy, env, macro, seed_base, device, hidden_size,
               frame_skip=8, max_actions=2250):
    """Deterministic argmax run on fixed seeds; returns a stats dict.

    The platform score is the median of per-run max distances, so this is the
    closest cheap proxy available during training. Additionally reports the
    mean/max, the share of runs that survived to the time limit, and the
    median energy fraction and speed at run end (diagnostics: dying with
    empty battery means poor energy management; dying at high speed on rough
    ground means control problems).
    """
    num_envs = env.num_envs
    memory = torch.zeros(num_envs, hidden_size, device=device)
    best_x = np.full(num_envs, 1.0, dtype=np.float32)
    done_flags = np.zeros(num_envs, dtype=bool)
    env.reset(seed_base)
    for i in range(1, num_envs):
        env.reset_at(i, seed=seed_base + i * 7919)
    memory.zero_()
    best_x[:] = env.obs[:, 0] * 1000.0
    done_flags[:] = False
    prev_action = torch.zeros(num_envs, dtype=torch.long, device=device)
    prev_reward = torch.zeros(num_envs, device=device)
    prev_done = torch.zeros(num_envs, device=device)
    progress = torch.zeros(num_envs, device=device)
    trial_start = torch.ones(num_envs, device=device)
    budget = env.trial_time_limit * 60
    steps = np.zeros(num_envs, dtype=np.int32)
    end_energy = np.zeros(num_envs, dtype=np.float32)
    end_speed = np.zeros(num_envs, dtype=np.float32)
    with torch.no_grad():
        for _ in range(max_actions):
            obs = torch.from_numpy(env.obs.copy()).to(device)
            logits, next_memory = policy(
                obs, prev_action, prev_reward, prev_done, progress, trial_start, memory)
            memory = next_memory
            action = logits.argmax(dim=-1)
            controls = macro[action.cpu().numpy()].copy()
            rewards = np.zeros(num_envs, dtype=np.float32)
            for _ in range(frame_skip):
                alive = ~done_flags
                if not alive.any():
                    break
                _, step_reward, term, trunc, _ = env.step_uint8(controls)
                rewards[alive] += step_reward[alive]
                cur_x = env.obs[:, 0] * 1000.0
                np.maximum(best_x, cur_x, out=best_x, where=alive)
                steps[alive] += 1
                done_flags |= term.astype(bool) | trunc.astype(bool)
                controls[done_flags] = 0
            end_energy[:] = env.obs[:, 6] * 100.0
            end_speed[:] = env.obs[:, 2] * 20.0
            progress = torch.from_numpy(
                np.minimum(steps / budget, 1.0).astype(np.float32)).to(device)
            prev_action = action
            prev_reward = torch.from_numpy(rewards).to(device)
            prev_done = torch.from_numpy(done_flags.astype(np.float32)).to(device)
            trial_start = torch.zeros(num_envs, device=device)
            if done_flags.all():
                break
    distances = np.maximum(0.0, best_x - 1.0)
    return {
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
        "survived": float(np.mean(~done_flags)),
        "energy": float(np.median(end_energy)),
        "speed": float(np.median(end_speed)),
    }


def save_checkpoint(path: Path, model_state, optimizer_state, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model_state, "optimizer": optimizer_state, "frames": frames}, path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-frames", type=int, default=4_000_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--extra-actions", type=str, default="",
                        help="Comma-separated macros to explore after the drive bootstrap, or 'all'.")
    parser.add_argument("--extra-logit-bias", type=float, default=-3.0,
                        help="Initial logit penalty for extra macros (default: -3).")
    parser.add_argument("--seed", type=int, default=int(
        os.environ.get("ARENA_SEED", "2026")))
    parser.add_argument("--max-seconds", type=int, default=6600)
    parser.add_argument("--save-interval", type=float, default=60.0)
    parser.add_argument("--eval-interval", type=float, default=300.0)
    parser.add_argument("--eval-envs", type=int, default=48)
    parser.add_argument("--stress-train-fraction", type=float, default=0.5,
                        help="Fraction of parallel training worlds split between dense and weather profiles.")
    parser.add_argument("--config", type=str, default="",
                        help="Optional path to a custom env.yaml (experiments).")
    parser.add_argument("--eval-hard-config", type=str, default="",
                        help="Dense-obstacle holdout config (defaults to bundled stress profile).")
    parser.add_argument("--eval-weather-config", type=str, default="",
                        help="Weather-stack holdout config (defaults to bundled stress profile).")
    parser.add_argument("--eval-ultra-config", type=str, default="",
                        help="Ultra holdout config (dense + weather; defaults to bundled stress profile).")
    default_output = Path("/output/policy.onnx") if Path("/output").is_dir() else (
        Path(__file__).resolve().parent / "artifacts" / "policy.onnx"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--checkpoint", type=Path, default=(
        Path("/tmp/training/latest.pt") if Path("/output").is_dir() else
        Path(__file__).resolve().parent / "artifacts" / "latest.pt"
    ))
    args = parser.parse_args()
    if args.extra_actions.strip().lower() == "all":
        extra_actions = tuple(action for action in range(len(ACTION_MACROS))
                              if action not in (1, 10, 11))
    else:
        try:
            extra_actions = tuple(sorted(set(
                int(value.strip()) for value in args.extra_actions.split(",") if value.strip()
            )))
        except ValueError:
            parser.error("--extra-actions must be comma-separated integers or 'all'")
    if any(not 0 <= action < len(ACTION_MACROS) for action in extra_actions):
        parser.error("--extra-actions contains a macro outside 0..30")
    if not np.isfinite(args.extra_logit_bias) or args.extra_logit_bias > 0:
        parser.error("--extra-logit-bias must be finite and <= 0")
    if not 0.0 <= args.stress_train_fraction < 1.0:
        parser.error("--stress-train-fraction must be in [0, 1)")
    if min(args.num_envs, args.rollout_steps, args.frame_skip, args.epochs,
           args.total_frames, args.max_seconds, args.hidden_size) <= 0:
        parser.error("all counts and durations must be positive")
    import mars_rover_env as _env_pkg
    config_dir = Path(_env_pkg.__file__).resolve().parent / "configs"
    if not args.eval_hard_config:
        candidate = config_dir / "eval_stress_dense.yaml"
        if candidate.exists():
            args.eval_hard_config = str(candidate)
    if not args.eval_weather_config:
        candidate = config_dir / "eval_stress_weather.yaml"
        if candidate.exists():
            args.eval_weather_config = str(candidate)
    if not args.eval_ultra_config:
        candidate = config_dir / "eval_stress_ultra.yaml"
        if candidate.exists():
            args.eval_ultra_config = str(candidate)

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # The policy is a tiny recurrent net: GPU transfers cost more than they
    # save, CPU training measures faster for this workload.
    device = torch.device("cpu")

    def training_config(path):
        config = load_env_config(path or None)
        config.reward.energy_cost_scale = 0.025
        config.reward.flip_penalty = 8.0
        config.reward.hard_contact_penalty = 0.0  # original penalty repeats cumulative damage every frame
        config.reward.stuck_penalty = 0.08
        config.biome_split = 0  # mix every public biome and mechanism stack
        return config

    stress_total = min(args.num_envs - 1,
                       round(args.num_envs * args.stress_train_fraction))
    available_stress = [path for path in (args.eval_hard_config, args.eval_weather_config,
                                          args.eval_ultra_config)
                        if path]
    if not available_stress:
        stress_total = 0
    stress_counts = [stress_total // len(available_stress)
                     + (i < stress_total % len(available_stress))
                     for i in range(len(available_stress))]
    standard_count = args.num_envs - stress_total
    cohorts = [MarsRoverVecEnv(standard_count,
                              config_override=training_config(args.config))]
    for path, count in zip(available_stress, stress_counts):
        if count:
            cohorts.append(MarsRoverVecEnv(
                count, config_override=training_config(path)))
    env = MixedVecEnv(cohorts) if len(cohorts) > 1 else cohorts[0]
    print(f"training cohorts: standard={standard_count} "
          f"stress={dict(zip(available_stress, stress_counts))}", flush=True)
    if env.obs_dim != 160 or len(ACTION_MACROS) != 31:
        raise RuntimeError("The evaluator observation/action contract changed")
    eval_config = load_env_config(args.config or None)
    eval_config.biome_split = 0  # hold out mixed worlds from the full public biome bank
    eval_env = MarsRoverVecEnv(args.eval_envs, config_override=eval_config)
    validation_env = MarsRoverVecEnv(args.eval_envs, config_override=eval_config)
    eval_macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    hard_env = None
    if args.eval_hard_config:
        hard_env = MarsRoverVecEnv(
            max(8, args.eval_envs), config_path=args.eval_hard_config)
        print(f"dense stress eval env: {args.eval_hard_config}", flush=True)
    weather_env = None
    if args.eval_weather_config:
        weather_env = MarsRoverVecEnv(
            max(8, args.eval_envs), config_path=args.eval_weather_config)
        print(f"weather stress eval env: {args.eval_weather_config}", flush=True)
    ultra_env = None
    if args.eval_ultra_config:
        ultra_env = MarsRoverVecEnv(
            max(8, args.eval_envs), config_path=args.eval_ultra_config)
        print(f"ultra stress eval env: {args.eval_ultra_config}", flush=True)
    policy = Policy(env.obs_dim, len(ACTION_MACROS), args.hidden_size,
                    extra_actions=extra_actions,
                    extra_logit_bias=args.extra_logit_bias).to(device)
    print(f"trainable macros={sorted({1, 10, 11, *extra_actions})} "
          f"extra_logit_bias={args.extra_logit_bias}", flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    zeros = lambda: torch.zeros(args.num_envs, device=device)
    obs = env.reset(args.seed).copy()
    state = {
        "obs": obs,
        "prev_action": torch.zeros(args.num_envs, dtype=torch.long, device=device),
        "prev_reward": zeros(),
        "prev_done": zeros(),
        "progress": zeros(),
        "trial_start": torch.ones(args.num_envs, device=device),
        "memory": torch.zeros(args.num_envs, args.hidden_size, device=device),
        "steps": np.zeros(args.num_envs, dtype=np.int32),
        "best": obs[:, 0].copy() * 1000.0,
    }
    cfg = argparse.Namespace(
        num_envs=args.num_envs, rollout_steps=args.rollout_steps,
        frame_skip=args.frame_skip, epochs=args.epochs, envs_per_batch=32,
        gamma=0.995, gae_lambda=0.95, clip=0.2, value_coef=0.5,
        entropy_coef=0.02, max_grad_norm=0.5,
    )
    start = time.monotonic()
    frames = 0
    update = 0
    last_save = -1e9
    last_eval = -1e9
    last_beat = -1e9
    rule_policy = RulePolicy()
    rule_stats = quick_eval(rule_policy, eval_env, eval_macro,
                            args.seed + 100_000, device, rule_policy.hidden_size)
    rule_validation = quick_eval(rule_policy, validation_env, eval_macro,
                                 args.seed + 300_000, device, rule_policy.hidden_size)
    rule_hard = (quick_eval(rule_policy, hard_env, eval_macro,
                            args.seed + 200_000, device, rule_policy.hidden_size)
                 if hard_env is not None else None)
    rule_weather = (quick_eval(rule_policy, weather_env, eval_macro,
                               args.seed + 400_000, device, rule_policy.hidden_size)
                    if weather_env is not None else None)
    rule_ultra = (quick_eval(rule_policy, ultra_env, eval_macro,
                             args.seed + 500_000, device, rule_policy.hidden_size)
                  if ultra_env is not None else None)
    hard_floor = rule_hard["median"] * 0.95 if rule_hard else 0.0
    weather_floor = rule_weather["median"] * 0.95 if rule_weather else 0.0
    ultra_floor = rule_ultra["median"] * 0.95 if rule_ultra else 0.0
    best_median = min(rule_stats["median"], rule_validation["median"])
    print(f"rule baseline: mixed_eval={rule_stats['median']:.1f}m "
          f"mixed_validation={rule_validation['median']:.1f}m "
          f"dense={(rule_hard or {}).get('median', float('nan')):.1f}m "
          f"weather={(rule_weather or {}).get('median', float('nan')):.1f}m "
          f"ultra={(rule_ultra or {}).get('median', float('nan')):.1f}m",
          flush=True)
    best_state = None
    best_frames = 0
    diagnostics = []
    recent_distances = []
    recent_action_counts = np.zeros(len(ACTION_MACROS), dtype=np.int64)
    diagnostics_path = args.checkpoint.with_name("diagnostics.json")
    print(
        f"SOLUTION={SOLUTION_NAME} seed={args.seed} hidden={args.hidden_size} "
        f"envs={args.num_envs} rollout={args.rollout_steps} skip={args.frame_skip} "
        f"epochs={args.epochs} max_seconds={args.max_seconds} "
        f"started={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        flush=True,
    )
    print(f"training on {device}", flush=True)
    if not export_policy(rule_policy, args.output):
        raise SystemExit("fallback policy export failed")
    while frames < args.total_frames and time.monotonic() - start < args.max_seconds:
        elapsed = time.monotonic() - start
        progress = min(1.0, elapsed / max(1.0, float(args.max_seconds)))
        cfg.entropy_coef = 0.005 - 0.004 * progress
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 - 2.0e-4 * progress

        batch, bootstrap, distances = collect(env, policy, state, cfg, device, rng)
        actor, critic, entropy = update_policy(
            policy, optimizer, batch, bootstrap, cfg, rng
        )
        recent_action_counts += np.bincount(
            batch["action"].detach().cpu().numpy().reshape(-1),
            minlength=len(ACTION_MACROS),
        )
        frames += args.num_envs * args.rollout_steps * args.frame_skip
        update += 1

        if not is_finite(policy):
            print("WARNING: weights diverged to NaN, rolling back to last good state",
                  flush=True)
            if best_state is not None:
                policy.load_state_dict(copy.deepcopy(best_state))
            else:
                raise SystemExit("weights diverged before any good checkpoint")

        if elapsed - last_eval >= args.eval_interval:
            last_eval = elapsed
            try:
                stats = quick_eval(
                    policy, eval_env, eval_macro, args.seed + 100_000,
                    device, args.hidden_size,
                )
                validation_stats = quick_eval(
                    policy, validation_env, eval_macro, args.seed + 300_000,
                    device, args.hidden_size,
                )
                stats["validation_median"] = round(validation_stats["median"], 1)
                selection_median = min(stats["median"], validation_stats["median"])
                stats["selection_median"] = round(selection_median, 1)
                stats["frames"] = frames
                stats["elapsed"] = round(elapsed, 1)
                hard_line = f" validation={validation_stats['median']:.1f}m"
                hard_acceptable = hard_env is None
                if hard_env is not None:
                    try:
                        hard_stats = quick_eval(
                            policy, hard_env, eval_macro, args.seed + 200_000,
                            device, args.hidden_size,
                        )
                        stats["dense_median"] = round(hard_stats["median"], 1)
                        hard_acceptable = hard_stats["median"] >= hard_floor
                        hard_line = (f" dense_median={hard_stats['median']:.1f}m "
                                     f"dense_survived={hard_stats['survived']:.0%}")
                    except Exception as exc:
                        hard_line = f" (hard eval failed: {exc})"
                weather_acceptable = weather_env is None
                if weather_env is not None:
                    try:
                        weather_stats = quick_eval(
                            policy, weather_env, eval_macro, args.seed + 400_000,
                            device, args.hidden_size,
                        )
                        stats["weather_median"] = round(weather_stats["median"], 1)
                        weather_acceptable = weather_stats["median"] >= weather_floor
                        hard_line += f" weather_median={weather_stats['median']:.1f}m"
                    except Exception as exc:
                        hard_line += f" (weather eval failed: {exc})"
                ultra_acceptable = ultra_env is None
                if ultra_env is not None:
                    try:
                        ultra_stats = quick_eval(
                            policy, ultra_env, eval_macro, args.seed + 500_000,
                            device, args.hidden_size,
                        )
                        stats["ultra_median"] = round(ultra_stats["median"], 1)
                        ultra_acceptable = ultra_stats["median"] >= ultra_floor
                        hard_line += f" ultra_median={ultra_stats['median']:.1f}m"
                    except Exception as exc:
                        hard_line += f" (ultra eval failed: {exc})"
                diagnostics.append(stats)
                if selection_median > best_median and hard_acceptable and weather_acceptable and ultra_acceptable:
                    best_median = selection_median
                    best_frames = frames
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
                    print(
                        f"BEST updated: eval median={stats['median']:.1f}m "
                        f"mean={stats['mean']:.1f}m max={stats['max']:.1f}m "
                        f"survived={stats['survived']:.0%} "
                        f"energy={stats['energy']:.0f}% speed={stats['speed']:.1f}m/s "
                        f"frames={frames}{hard_line}", flush=True,
                    )
                else:
                    print(
                        f"eval median={stats['median']:.1f}m mean={stats['mean']:.1f}m "
                        f"max={stats['max']:.1f}m survived={stats['survived']:.0%} "
                        f"energy={stats['energy']:.0f}% speed={stats['speed']:.1f}m/s "
                        f"frames={frames} best={best_median:.1f}m{hard_line}", flush=True,
                    )
            except Exception as exc:
                print(f"WARNING: eval failed: {exc}", flush=True)

        if elapsed - last_save >= args.save_interval:
            last_save = elapsed
            # /output/policy.onnx always holds the transferable rule fallback.
            try:
                export_policy(rule_policy, args.output)
            except Exception as exc:
                print(f"WARNING: export failed: {exc}", flush=True)
            save_checkpoint(args.checkpoint, policy.state_dict(),
                            optimizer.state_dict(), frames)
            try:
                import json
                with open(diagnostics_path, "w") as handle:
                    json.dump(diagnostics, handle, indent=1)
            except Exception:
                pass

        recent_distances.extend(distances)
        if update % 50 == 0:
            train_median = (f"{np.median(recent_distances):.1f}m"
                            if recent_distances else "n/a")
            elapsed_now = time.monotonic() - start
            print(
                f"update={update} frames={frames} "
                f"episodes_last_50={len(recent_distances)} "
                f"train_median_last_50={train_median} actor={actor:.3f} "
                f"critic={critic:.3f} entropy={entropy:.3f} "
                f"elapsed={elapsed_now:.0f}s "
                f"fps={frames / max(1.0, elapsed_now):.0f} "
                f"action_counts_last_50="
                f"{ {i: int(n) for i, n in enumerate(recent_action_counts) if n} }",
                flush=True,
            )
            recent_distances.clear()
            recent_action_counts.fill(0)

    # Platform evidence: the residual PPO does not transfer to the organizer
    # build (in-training proxy ~413 m -> 10.5 m on the test), while the
    # deterministic rule controller is the most transferable artifact.
    # Always export the fallback, regardless of the training outcome.
    if not export_policy(rule_policy, args.output):
        raise SystemExit("final fallback export failed")
    print("final fallback export (rule controller)", flush=True)
    save_checkpoint(args.checkpoint, policy.state_dict(),
                    optimizer.state_dict(), frames)
    elapsed_total = time.monotonic() - start
    print(
        f"SUMMARY frames={frames} updates={update} "
        f"elapsed={elapsed_total:.0f}s fps={frames / max(1.0, elapsed_total):.0f} "
        f"best_median={best_median:.1f}m best_frames={best_frames}",
        flush=True,
    )
    try:
        import json
        with open(diagnostics_path, "w") as handle:
            json.dump(diagnostics, handle, indent=1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
