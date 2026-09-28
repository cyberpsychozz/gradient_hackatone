from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from mars_rover_env.actions import ACTION_MACROS
from model import LinearPolicy
from train import load_params


class SubmissionSmokeTest(unittest.TestCase):
    def test_policy_shapes(self) -> None:
        batch = 4
        policy = LinearPolicy(obs_dim=160, action_dim=len(ACTION_MACROS))
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

    def test_load_params(self) -> None:
        policy = LinearPolicy(obs_dim=160, action_dim=len(ACTION_MACROS))
        params = np.zeros(policy.action_dim * policy.feature_dim + policy.action_dim)
        load_params(policy, params)
        self.assertTrue(np.allclose(
            policy.linear.weight.detach().numpy().reshape(-1), params[:policy.action_dim * policy.feature_dim]))

    def test_export_matches_evaluator_format(self) -> None:
        from check_policy import validate_policy
        from train import export_policy

        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            export_policy(LinearPolicy(160, len(ACTION_MACROS)), path)
            validate_policy(path)

    def test_training_entrypoint_has_expected_functions(self) -> None:
        import train

        self.assertTrue(callable(train.evaluate))
        self.assertTrue(callable(train.load_params))
        self.assertTrue(callable(train.export_policy))


if __name__ == "__main__":
    unittest.main()
