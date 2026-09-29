"""Train a linear rover policy with the Cross-Entropy Method (CEM)."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS
from model import LinearPolicy


def export_policy(model: LinearPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = copy.deepcopy(model).cpu().eval()
    try:
        from arena.protocol import export_onnx
    except ModuleNotFoundError as exc:
        if exc.name != "arena":
            raise
        # The local environment lacks arena-base. The server uses its validator.
        import onnx
        from onnx import TensorProto, helper as onnx_helper

        n = 2
        example = (
            torch.zeros(n, module.obs_dim), torch.zeros(n, dtype=torch.int64),
            torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.ones(n),
            torch.zeros(n, module.memory_size),
        )
        inputs = (
            "observation", "previous_action", "previous_reward",
            "previous_done", "trial_progress", "trial_start", "memory",
        )
        torch.onnx.export(
            module, example, str(path), opset_version=17, dynamo=False,
            input_names=list(inputs), output_names=["logits", "next_memory"],
            dynamic_axes={name: {0: "batch"} for name in (*inputs, "logits", "next_memory")},
        )
        graph = onnx.load(str(path))

        def ensure_contract_inputs(graph) -> None:
            expected = (
                "observation", "previous_action", "previous_reward", "previous_done",
                "trial_progress", "trial_start", "memory",
            )
            existing = {item.name for item in graph.graph.input}
            if existing == set(expected):
                return
            dtypes = {"previous_action": TensorProto.INT64}
            by_name = {item.name: item for item in graph.graph.input}
            for name in expected:
                if name not in existing:
                    value_info = onnx_helper.make_tensor_value_info(
                        name, dtypes.get(name, TensorProto.FLOAT), None)
                    value_info.type.tensor_type.shape.dim.add().dim_param = "batch"
                    by_name[name] = value_info
            del graph.graph.input[:]
            for name in expected:
                graph.graph.input.append(by_name[name])

        ensure_contract_inputs(graph)
        entry = graph.metadata_props.add()
        entry.key, entry.value = "rover.format", "rover-policy-onnx-v1"
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
    else:
        export_onnx(str(path), module, obs_dim=module.obs_dim,
                    memory_size=module.memory_size)
    print(f"exported {path} ({path.stat().st_size:,} bytes)", flush=True)


def load_params(policy: LinearPolicy, params: np.ndarray) -> None:
    weight_size = policy.action_dim * policy.feature_dim
    weight = params[:weight_size].reshape(policy.action_dim, policy.feature_dim)
    bias = params[weight_size:]
    with torch.no_grad():
        policy.linear.weight.copy_(torch.from_numpy(weight).float())
        policy.linear.bias.copy_(torch.from_numpy(bias).float())


def build_features(obs, prev_action, prev_reward, prev_done, progress,
                   action_dim, device):
    one_hot = torch.nn.functional.one_hot(
        torch.from_numpy(prev_action).to(device).long(), action_dim).float()
    return torch.cat((
        torch.from_numpy(obs).to(device).clamp(-10.0, 10.0),
        one_hot,
        torch.tanh(torch.from_numpy(prev_reward).to(device).unsqueeze(-1) / 10),
        torch.from_numpy(prev_done).to(device).unsqueeze(-1),
        torch.from_numpy(progress).to(device).unsqueeze(-1),
    ), dim=-1)


def evaluate(env, candidates, cfg, device, macro, budget, rng):
    """Run each candidate policy on one env for `eval_steps` decisions.

    Returns the per-candidate best distance in metres (clamped at zero).
    """
    population = env.num_envs
    candidates = candidates.astype(np.float32)
    weight_size = cfg.action_dim * cfg.feature_dim
    weights = torch.from_numpy(
        candidates[:, :weight_size].reshape(population, cfg.action_dim,
                                            cfg.feature_dim)).to(device)
    biases = torch.from_numpy(candidates[:, weight_size:]).to(device)

    env.reset(int(rng.integers(0, 2**31)))
    obs = env.obs.copy()
    prev_action = np.zeros(population, dtype=np.int64)
    prev_reward = np.zeros(population, dtype=np.float32)
    prev_done = np.zeros(population, dtype=np.float32)
    progress = np.zeros(population, dtype=np.float32)
    steps = np.zeros(population, dtype=np.int32)
    best = obs[:, 0].copy() * 1000.0
    done = np.zeros(population, dtype=bool)

    for _ in range(cfg.eval_steps):
        features = build_features(
            obs, prev_action, prev_reward, prev_done, progress,
            cfg.action_dim, device,
        )
        logits = torch.einsum("nd,ned->ne", features, weights) + biases
        action = logits.argmax(1).cpu().numpy()
        controls = macro[action].copy()
        controls[done] = 0
        rewards = np.zeros(population, dtype=np.float32)
        for _ in range(cfg.frame_skip):
            _, step_reward, terminated, truncated, _ = env.step_uint8(controls)
            active = ~done
            rewards[active] += step_reward[active]
            best[active] = np.maximum(best[active], env.obs[active, 0] * 1000.0)
            done |= terminated.astype(bool) | truncated.astype(bool)
            controls[done] = 0
            if done.all():
                break
        prev_action = action
        prev_reward = rewards
        prev_done = done.astype(np.float32)
        steps[~done] += cfg.frame_skip
        progress = np.minimum(steps / budget, 1.0).astype(np.float32)
        obs = env.obs.copy()
        if done.all():
            break
    return np.maximum(best - 1.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--eval-steps", type=int, default=450)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--noise-floor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=int(
        os.environ.get("ARENA_SEED", "2026")))
    parser.add_argument("--max-seconds", type=int, default=6900)
    default_output = Path("/output/policy.onnx") if Path("/output").is_dir() else (
        Path(__file__).resolve().parent / "artifacts" / "policy.onnx"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    if min(args.pop_size, args.generations, args.eval_steps, args.frame_skip) <= 0:
        parser.error("all counts and durations must be positive")
    if not 0.0 < args.elite_fraction <= 1.0:
        parser.error("elite-fraction must be in (0, 1]")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MarsRoverVecEnv(args.pop_size)
    action_dim = len(ACTION_MACROS)
    feature_dim = env.obs_dim + action_dim + 3
    if env.obs_dim != 160 or action_dim != 31:
        raise RuntimeError("The evaluator observation/action contract changed")
    macro = np.asarray(ACTION_MACROS, dtype=np.int32)
    budget = max(1, round(env.trial_time_limit * 60))

    params_size = action_dim * feature_dim + action_dim
    mean = np.zeros(params_size)
    std = np.full(params_size, args.sigma)
    elite_count = max(1, round(args.pop_size * args.elite_fraction))
    best_params = np.zeros(params_size)
    best_score = -np.inf
    start = time.monotonic()
    print(f"training on {device}", flush=True)

    cfg = argparse.Namespace(
        action_dim=action_dim, feature_dim=feature_dim, eval_steps=args.eval_steps,
        frame_skip=args.frame_skip,
    )
    for generation in range(args.generations):
        if time.monotonic() - start >= args.max_seconds:
            break
        candidates = rng.normal(mean, std, size=(args.pop_size, params_size))
        scores = evaluate(env, candidates, cfg, device, macro, budget, rng)
        order = np.argsort(scores)[::-1]
        elites = candidates[order[:elite_count]]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + args.noise_floor
        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
            best_params = candidates[order[0]].copy()
        print(
            f"generation={generation + 1} best={scores[order[0]]:.1f}m "
            f"median={np.median(scores):.1f}m elite_mean={scores[order[:elite_count]].mean():.1f}m "
            f"alltime_best={best_score:.1f}m elapsed={time.monotonic() - start:.0f}s",
            flush=True,
        )

    policy = LinearPolicy(env.obs_dim, action_dim)
    load_params(policy, best_params)
    export_policy(policy, args.output)
    print(f"best score {best_score:.1f}m", flush=True)


if __name__ == "__main__":
    main()
