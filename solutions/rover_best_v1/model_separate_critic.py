"""Recurrent PPO actor with a separate feedforward value function.

The critic consumes raw features and cannot change the actor encoder or GRU.
This is an actor/critic architecture variant, not a full PPG implementation.
"""
from __future__ import annotations

import math

from torch import nn

from model_residual import Policy as SharedPolicy


class Policy(SharedPolicy):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int,
                 extra_actions: tuple[int, ...] = (), extra_logit_bias: float = -3.0):
        super().__init__(obs_dim, action_dim, hidden_size,
                         extra_actions=extra_actions,
                         extra_logit_bias=extra_logit_bias)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def value(self, features, memory):
        return self.critic(features).squeeze(-1)
