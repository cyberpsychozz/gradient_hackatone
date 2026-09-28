from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_INPUTS = (
    "observation", "previous_action", "previous_reward", "previous_done",
    "trial_progress", "trial_start", "memory",
)
EXPECTED_OUTPUTS = ("logits", "next_memory")
FORMAT_KEY = "rover.format"
FORMAT_VALUE = "rover-policy-onnx-v1"


def validate_policy(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"policy artifact is missing or empty: {path}")

    import onnx

    graph = onnx.load(str(path))
    if not any(
        prop.key == FORMAT_KEY and prop.value == FORMAT_VALUE
        for prop in graph.metadata_props
    ):
        raise RuntimeError(f"missing {FORMAT_KEY}={FORMAT_VALUE} metadata")
    if tuple(item.name for item in graph.graph.input) != EXPECTED_INPUTS:
        raise RuntimeError("ONNX inputs do not match the rover policy contract")
    if tuple(item.name for item in graph.graph.output) != EXPECTED_OUTPUTS:
        raise RuntimeError("ONNX outputs do not match the rover policy contract")

    memory_size = 0
    for item in graph.graph.input:
        if item.name == "memory":
            shape = item.type.tensor_type.shape
            memory_size = shape.dim[1].dim_value if len(shape.dim) > 1 else 0
    if memory_size < 1:
        raise RuntimeError("ONNX memory input has an invalid size")

    onnx.checker.check_model(graph)

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print(
            f"policy artifact is valid (graph only): {path} "
            f"({path.stat().st_size} bytes)"
        )
        return

    import numpy as np

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if tuple(item.name for item in session.get_inputs()) != EXPECTED_INPUTS:
        raise RuntimeError("ONNX inputs do not match the rover policy contract")
    if tuple(item.name for item in session.get_outputs()) != EXPECTED_OUTPUTS:
        raise RuntimeError("ONNX outputs do not match the rover policy contract")
    if session.get_modelmeta().custom_metadata_map.get(FORMAT_KEY) != FORMAT_VALUE:
        raise RuntimeError(f"missing {FORMAT_KEY} metadata")

    batch = 48
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
