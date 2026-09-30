"""Two-parameter controller optimized against complete simulated races."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):
    obs_dim = 160
    hidden_size = 1

    def __init__(self, pitch_threshold: float = 0.1,
                 angular_velocity_gain: float = 0.0):
        super().__init__()
        self.register_buffer("pitch_threshold", torch.tensor(float(pitch_threshold)))
        self.register_buffer("angular_velocity_gain", torch.tensor(float(angular_velocity_gain)))

    def forward(self, observation, previous_action, previous_reward,
                previous_done, trial_progress, trial_start, memory):
        # Observation 5 is angular velocity divided by ten.
        tilt = observation[:, 4] + self.angular_velocity_gain * observation[:, 5] * 10.0
        context = 1e-8 * (
            observation[:, 0] + previous_action.float() + previous_reward
            + previous_done + trial_progress + memory[:, 0]
        )
        drive = previous_action * 0 + 1
        drive = torch.where(tilt > self.pitch_threshold,
                            previous_action * 0 + 11, drive)
        drive = torch.where(tilt < -self.pitch_threshold,
                            previous_action * 0 + 10, drive)
        phase = memory[:, 0]
        action = torch.where(phase < 2.5, previous_action * 0 + 13, drive)
        action = torch.where((phase >= 0.5) & (phase < 1.5),
                             previous_action * 0 + 1, action)
        action = torch.where(trial_start > 0.5,
                             previous_action * 0 + 13, action)
        logits = F.one_hot(action, num_classes=31).float() * 100.0
        logits = logits + context.unsqueeze(-1)
        next_memory = torch.where(trial_start.unsqueeze(-1) > 0.5,
                                  memory * 0.0 + 1.0,
                                  (memory + 1.0).clamp(max=3.0))
        return logits, next_memory
