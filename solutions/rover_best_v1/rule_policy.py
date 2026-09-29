"""Weight-free fallback controller: engage all-wheel drive, then level the body."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RulePolicy(nn.Module):
    obs_dim = 160
    hidden_size = 1
    level_threshold = 0.1

    def forward(self, observation, previous_action, previous_reward,
                previous_done, trial_progress, trial_start, memory):
        # The graph keeps all seven protocol inputs; context is far too small
        # to change argmax, but prevents ONNX export from pruning them.
        context = 1e-8 * (
            observation[:, 0] + previous_action.float() + previous_reward
            + previous_done + trial_progress + memory[:, 0]
        )
        drive = previous_action * 0 + 1
        drive = torch.where(observation[:, 4] > self.level_threshold,
                            previous_action * 0 + 11, drive)
        drive = torch.where(observation[:, 4] < -self.level_threshold,
                            previous_action * 0 + 10, drive)
        phase = memory[:, 0]
        # Toggle drive 0 -> 1, release the toggle, toggle 1 -> 2.
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
