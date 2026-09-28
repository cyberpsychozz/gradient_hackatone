from __future__ import annotations

import argparse
from pathlib import Path


def validate_policy(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"policy artifact is missing or empty: {path}")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected_inputs = (
        "observation", "previous_action", "previous_reward", "previous_done",
        "trial_progress", "trial_start", "memory",
    )
    if tuple(item.name for item in session.get_inputs()) != expected_inputs:
        raise RuntimeError("ONNX inputs do not match the rover policy contract")
    if tuple(item.name for item in session.get_outputs()) != ("logits", "next_memory"):
        raise RuntimeError("ONNX outputs do not match the rover policy contract")
    if session.get_modelmeta().custom_metadata_map.get("rover.format") != "rover-policy-onnx-v1":
        raise RuntimeError("missing rover.format metadata")
    batch = 48
    memory_size = session.get_inputs()[-1].shape[1]
    feeds = {
        "observation": np.zeros((batch, 160), dtype=np.float32),
        "previous_action": np.zeros(batch, dtype=np.int64),
        "previous_reward": np.zeros(batch, dtype=np.float32),
        "previous_done": np.zeros(batch, dtype=np.float32),
        "trial_progress": np.zeros(batch, dtype=np.float32),
        "trial_start": np.ones(batch, dtype=np.float32),
        "memory": np.zeros((batch, memory_size), dtype=np.float32),
    }
    logits, next_memory = session.run(None, feeds)
    if logits.shape != (batch, 31) or next_memory.shape != (batch, memory_size):
        raise RuntimeError("ONNX outputs have incorrect shapes")
    if not np.isfinite(logits).all() or not np.isfinite(next_memory).all():
        raise RuntimeError("ONNX outputs contain non-finite values")
    print(f"policy artifact is valid: {path} ({path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("/output/policy.onnx"))
    args = parser.parse_args()
    validate_policy(args.path)


if __name__ == "__main__":
    main()
