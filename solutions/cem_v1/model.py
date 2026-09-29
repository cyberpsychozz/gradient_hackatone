from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearPolicy(nn.Module):
    """Linear policy logits = features @ W^T + b. Memory size is 1."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_size = 1
        self.feature_dim = obs_dim + action_dim + 3
        self.linear = nn.Linear(self.feature_dim, action_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def features(self, observation, previous_action, previous_reward,
                 previous_done, trial_progress):
        encoded_action = F.one_hot(previous_action, self.action_dim).float()
        return torch.cat((
            observation.clamp(-10.0, 10.0),
            encoded_action,
            torch.tanh(previous_reward.unsqueeze(-1) / 10),
            previous_done.unsqueeze(-1),
            trial_progress.unsqueeze(-1),
        ), dim=-1)

    def forward(self, observation, previous_action, previous_reward, previous_done,
                trial_progress, trial_start, memory):
        logits = self.linear(self.features(
            observation, previous_action, previous_reward, previous_done, trial_progress))
        next_memory = memory * 0.0
        return logits, next_memory
