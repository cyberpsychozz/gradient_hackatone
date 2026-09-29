"""Run an exported rover-policy-onnx-v1 model in the student GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mars_rover_env.actions import ACTION_MACROS


class PolicyController:
    def __init__(self, path: str | Path, trial_seconds: float, frame_skip: int = 8):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX playback requires onnxruntime in the GUI's Python environment"
            ) from exc

        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"policy not found: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = {item.name: item for item in self.session.get_inputs()}
        expected = {
            "observation", "previous_action", "previous_reward", "previous_done",
            "trial_progress", "trial_start", "memory",
        }
        if set(inputs) != expected:
            raise ValueError(f"policy inputs differ from rover contract: {set(inputs) ^ expected}")
        memory_size = inputs["memory"].shape[1]
        if not isinstance(memory_size, int) or memory_size < 1:
            raise ValueError("policy memory must have a fixed positive width")
        self.memory_size = memory_size
        self.frame_skip = frame_skip
        self.frame_budget = max(1, round(trial_seconds * 60))
        self.reset()

    def reset(self) -> None:
        self.memory = np.zeros((1, self.memory_size), dtype=np.float32)
        self.previous_action = 0
        self.previous_reward = 0.0
        self.reward_sum = 0.0
        self.frames = 0
        self.hold_remaining = 0
        self.action_index = 0
        self.control = 0
        self.trial_start = True

    def action(self, observation: np.ndarray) -> int:
        if self.hold_remaining == 0:
            logits, memory = self.session.run(
                ["logits", "next_memory"],
                {
                    "observation": np.asarray(observation, dtype=np.float32).reshape(1, -1),
                    "previous_action": np.asarray([self.previous_action], dtype=np.int64),
                    "previous_reward": np.asarray([self.previous_reward], dtype=np.float32),
                    "previous_done": np.asarray([0.0], dtype=np.float32),
                    "trial_progress": np.asarray(
                        [min(self.frames / self.frame_budget, 1.0)], dtype=np.float32
                    ),
                    "trial_start": np.asarray([float(self.trial_start)], dtype=np.float32),
                    "memory": self.memory,
                },
            )
            if logits.shape != (1, len(ACTION_MACROS)) or not np.isfinite(logits).all():
                raise ValueError("policy returned invalid logits")
            if memory.shape != self.memory.shape or not np.isfinite(memory).all():
                raise ValueError("policy returned invalid memory")
            self.memory = memory
            self.action_index = int(np.argmax(logits[0]))
            self.control = ACTION_MACROS[self.action_index]
            self.previous_action = self.action_index
            self.previous_reward = 0.0
            self.reward_sum = 0.0
            self.hold_remaining = self.frame_skip
            self.trial_start = False
        return self.control

    def observe(self, reward: float) -> None:
        self.reward_sum += reward
        self.frames += 1
        self.hold_remaining -= 1
        if self.hold_remaining == 0:
            self.previous_reward = self.reward_sum
