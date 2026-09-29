from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Reinforce(nn.Module):
    """REINFORCE policy with a learned value baseline. Memory size is 1."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.memory_size = 1
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        for layer in (self.encoder[0], self.encoder[2]):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

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

    def evaluate(self, observation, previous_action, previous_reward,
                 previous_done, trial_progress):
        """Batched (over time) forward returning logits and value."""
        features = self.features(
            observation, previous_action, previous_reward, previous_done, trial_progress)
        hidden = self.encoder(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def step(self, observation, previous_action, previous_reward, previous_done,
             trial_progress, trial_start, memory):
        logits, value = self.evaluate(
            observation, previous_action, previous_reward, previous_done, trial_progress)
        next_memory = memory * 0.0
        return logits, value, next_memory

    def forward(self, observation, previous_action, previous_reward, previous_done,
                trial_progress, trial_start, memory):
        logits, _, next_memory = self.step(
            observation, previous_action, previous_reward, previous_done,
            trial_progress, trial_start, memory,
        )
        return logits, next_memory

    def sequence(self, observation, previous_action, previous_reward, previous_done,
                 trial_progress, trial_start, memory):
        logits = []
        values = []
        for index in range(len(observation)):
            current_logits, current_values, memory = self.step(
                observation[index], previous_action[index], previous_reward[index],
                previous_done[index], trial_progress[index], trial_start[index], memory,
            )
            logits.append(current_logits)
            values.append(current_values)
        return torch.stack(logits), torch.stack(values)
