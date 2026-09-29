from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int,
                 extra_actions: tuple[int, ...] = (), extra_logit_bias: float = -3.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 3, hidden_size),
            nn.Tanh(),
        )
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        if any(not 0 <= action < action_dim for action in extra_actions):
            raise ValueError("extra_actions contains an invalid macro")
        action_bias = torch.full((action_dim,), -1e4)
        if extra_actions:
            action_bias[list(extra_actions)] = extra_logit_bias
        action_bias[[1, 10, 11]] = 0.0
        self.register_buffer("action_bias", action_bias)
        for layer in (self.encoder[0], self.actor, self.critic):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def initial(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def step(self, observation, previous_action, previous_reward, previous_done,
             trial_progress, trial_start, memory):
        phase = memory[:, -1]
        memory = memory * (1 - trial_start.float().unsqueeze(-1))
        encoded_action = F.one_hot(previous_action, self.action_dim).float()
        features = torch.cat((
            observation.clamp(-10.0, 10.0),
            encoded_action,
            torch.tanh(previous_reward.unsqueeze(-1) / 10),
            previous_done.unsqueeze(-1),
            trial_progress.unsqueeze(-1),
        ), dim=-1)
        recurrent = self.memory(self.encoder(features), memory)
        next_phase = torch.where(trial_start > 0.5, phase * 0.0 + 1.0,
                                 (phase + 1.0).clamp(max=3.0))
        memory = torch.cat((recurrent[:, :-1], next_phase.unsqueeze(-1)), dim=-1)
        # The first three actions engage full drive, then PPO may correct pitch.
        preferred = previous_action * 0 + 1
        preferred = torch.where(observation[:, 4] > 0.1,
                                previous_action * 0 + 11, preferred)
        preferred = torch.where(observation[:, 4] < -0.1,
                                previous_action * 0 + 10, preferred)
        prior = F.one_hot(preferred, self.action_dim).float() * 3.0
        logits = (self.actor(memory) + self.action_bias + prior).clamp(-1.0e4, 1.0e4)
        scripted = torch.where((phase >= 0.5) & (phase < 1.5),
                               previous_action * 0 + 1,
                               previous_action * 0 + 13)
        scripted_logits = F.one_hot(scripted, self.action_dim).float() * 200.0 - 100.0
        scripted_step = ((phase < 2.5).float() + trial_start) > 0.5
        logits = torch.where(scripted_step.unsqueeze(-1), scripted_logits, logits)
        return logits, self.critic(memory).squeeze(-1), memory

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
