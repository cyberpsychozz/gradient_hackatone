from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """Feed-forward Double DQN. Stateless: memory size is 1."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.memory_size = 1
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        for layer in (self.net[0], self.net[2]):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[4].weight, 0.01)
        nn.init.zeros_(self.net[4].bias)

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

    def q_values(self, observation, previous_action, previous_reward,
                 previous_done, trial_progress):
        return self.net(self.features(
            observation, previous_action, previous_reward, previous_done, trial_progress))

    def forward(self, observation, previous_action, previous_reward, previous_done,
                trial_progress, trial_start, memory):
        next_memory = memory * 0.0
        logits = self.q_values(
            observation, previous_action, previous_reward, previous_done, trial_progress)
        return logits, next_memory
