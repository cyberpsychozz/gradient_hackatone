from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mars_rover_env.actions import ACTION_MACROS
from model import DQN


class SubmissionSmokeTest(unittest.TestCase):
    def test_policy_shapes(self) -> None:
        batch = 4
        policy = DQN(obs_dim=160, action_dim=len(ACTION_MACROS), hidden_size=32)
        logits, next_memory = policy(
            torch.zeros(batch, 160),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch),
            torch.zeros(batch),
            torch.zeros(batch),
            torch.ones(batch),
            torch.zeros(batch, 1),
        )
        self.assertEqual(tuple(logits.shape), (batch, len(ACTION_MACROS)))
        self.assertEqual(tuple(next_memory.shape), (batch, 1))
        self.assertEqual(policy.memory_size, 1)

    def test_export_matches_evaluator_format(self) -> None:
        from check_policy import validate_policy
        from train import export_policy

        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            export_policy(DQN(160, len(ACTION_MACROS), 32), path)
            validate_policy(path)

    def test_training_entrypoint_has_expected_functions(self) -> None:
        import train

        self.assertTrue(callable(train.select_actions))
        self.assertTrue(callable(train.update_dqn))
        self.assertTrue(callable(train.export_policy))
        self.assertTrue(callable(train.ReplayBuffer))


if __name__ == "__main__":
    unittest.main()
