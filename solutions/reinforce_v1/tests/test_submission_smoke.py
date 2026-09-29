from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from mars_rover_env.actions import ACTION_MACROS
from model import Reinforce
from train import discounted_returns


class SubmissionSmokeTest(unittest.TestCase):
    def test_policy_shapes(self) -> None:
        batch = 4
        steps = 3
        policy = Reinforce(obs_dim=160, action_dim=len(ACTION_MACROS), hidden_size=32)
        memory = torch.zeros(batch, 1)
        logits, values, next_memory = policy.step(
            torch.zeros(batch, 160),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch),
            torch.zeros(batch),
            torch.zeros(batch),
            torch.ones(batch),
            memory,
        )
        self.assertEqual(tuple(logits.shape), (batch, len(ACTION_MACROS)))
        self.assertEqual(tuple(values.shape), (batch,))
        self.assertEqual(tuple(next_memory.shape), (batch, 1))
        logits, values = policy.sequence(
            torch.zeros(steps, batch, 160),
            torch.zeros(steps, batch, dtype=torch.long),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            memory,
        )
        self.assertEqual(tuple(logits.shape), (steps, batch, len(ACTION_MACROS)))
        self.assertEqual(tuple(values.shape), (steps, batch))

    def test_discounted_returns(self) -> None:
        returns = discounted_returns([1.0, 1.0, 1.0], 0.5)
        self.assertTrue(np.allclose(returns, [1.75, 1.5, 1.0]))

    def test_export_matches_evaluator_format(self) -> None:
        from check_policy import validate_policy
        from train import export_policy

        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            export_policy(Reinforce(160, len(ACTION_MACROS), 32), path)
            validate_policy(path)

    def test_training_entrypoint_has_expected_functions(self) -> None:
        import train

        self.assertTrue(callable(train.collect_episodes))
        self.assertTrue(callable(train.finalize_episode))
        self.assertTrue(callable(train.update_policy))
        self.assertTrue(callable(train.export_policy))


if __name__ == "__main__":
    unittest.main()
