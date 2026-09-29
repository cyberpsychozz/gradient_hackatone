"""Vector environment made from independent simulator cohorts.

Each cohort has its own terrain/weather config. The combined observation and
step arrays retain the MarsRoverVecEnv interface used by recurrent PPO.
"""
from __future__ import annotations

import numpy as np


class MixedVecEnv:
    def __init__(self, cohorts):
        if not cohorts:
            raise ValueError("at least one simulator cohort is required")
        self.cohorts = tuple(cohorts)
        self.offsets = np.cumsum([0, *(env.num_envs for env in cohorts)])
        self.num_envs = int(self.offsets[-1])
        self.obs_dim = cohorts[0].obs_dim
        self.trial_time_limit = cohorts[0].trial_time_limit
        if any(env.obs_dim != self.obs_dim or
               env.trial_time_limit != self.trial_time_limit for env in cohorts):
            raise ValueError("simulator cohorts must share observation and trial contracts")
        self.obs = np.zeros((self.num_envs, self.obs_dim), np.float32)
        self.rewards = np.zeros(self.num_envs, np.float32)
        self.terminated = np.zeros(self.num_envs, np.uint8)
        self.truncated = np.zeros(self.num_envs, np.uint8)
        self.next_trial_start = np.zeros(self.num_envs, bool)

    def reset(self, seed=0):
        for group, env in enumerate(self.cohorts):
            start, stop = self.offsets[group:group + 2]
            self.obs[start:stop] = env.reset(seed + group * 1_000_003)
            self.next_trial_start[start:stop] = env.next_trial_start
        return self.obs

    def reset_at(self, env_id, seed=0, trial_start=True):
        if not 0 <= env_id < self.num_envs:
            raise IndexError(env_id)
        group = int(np.searchsorted(self.offsets, env_id, side="right") - 1)
        start = int(self.offsets[group])
        env = self.cohorts[group]
        self.obs[env_id] = env.reset_at(env_id - start, seed=seed, trial_start=trial_start)
        self.next_trial_start[env_id] = env.next_trial_start[env_id - start]
        return self.obs[env_id]

    def step_uint8(self, actions):
        for group, env in enumerate(self.cohorts):
            start, stop = self.offsets[group:group + 2]
            obs, rewards, terminated, truncated, _ = env.step_uint8(actions[start:stop])
            self.obs[start:stop] = obs
            self.rewards[start:stop] = rewards
            self.terminated[start:stop] = terminated
            self.truncated[start:stop] = truncated
            self.next_trial_start[start:stop] = env.next_trial_start
        return self.obs, self.rewards, self.terminated, self.truncated, {}
