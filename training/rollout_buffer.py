"""
training/rollout_buffer.py — On-policy rollout buffer with GAE

Stores transitions from num_envs parallel environments over rollout_steps,
then computes returns and advantages using Generalized Advantage Estimation.

Buffer layout (per field): [rollout_steps, num_envs, ...]
After compute_returns(): advantages and returns are ready for mini-batch sampling.
"""

import numpy as np
import torch
from typing import Generator, Tuple, Dict

from config import G1Config


class RolloutBuffer:
    """
    Stores a single rollout of experience across multiple parallel environments.
    Supports GAE advantage computation and random mini-batch iteration.
    """

    def __init__(self, cfg: G1Config, device: torch.device):
        self.cfg     = cfg
        self.device  = device
        self.T       = cfg.rollout_steps   # steps per env
        self.N       = cfg.num_envs        # parallel envs
        self.obs_dim = cfg.obs_dim
        self.act_dim = cfg.action_dim

        self._init_buffers()
        self.ptr = 0   # current write position (step index)
        self.full = False

    def _init_buffers(self):
        T, N = self.T, self.N
        self.obs       = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.actions   = np.zeros((T, N, self.act_dim), dtype=np.float32)
        self.rewards   = np.zeros((T, N),               dtype=np.float32)
        self.values    = np.zeros((T, N),               dtype=np.float32)
        self.log_probs = np.zeros((T, N),               dtype=np.float32)
        self.dones     = np.zeros((T, N),               dtype=np.float32)
        # Filled after compute_returns()
        self.advantages = np.zeros((T, N),              dtype=np.float32)
        self.returns    = np.zeros((T, N),              dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,   # (N, obs_dim)
        action:   np.ndarray,   # (N, act_dim)
        reward:   np.ndarray,   # (N,)
        value:    np.ndarray,   # (N,)
        log_prob: np.ndarray,   # (N,)
        done:     np.ndarray,   # (N,)
    ):
        assert self.ptr < self.T, "Buffer is full; call reset() before adding."
        t = self.ptr
        self.obs[t]       = obs
        self.actions[t]   = action
        self.rewards[t]   = reward
        self.values[t]    = value
        self.log_probs[t] = log_prob
        self.dones[t]     = done
        self.ptr += 1
        if self.ptr == self.T:
            self.full = True

    def compute_returns(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute GAE advantages and discounted returns.

        last_values: (N,) — V(s_{T}) bootstrap from critic
        last_dones:  (N,) — whether last state is terminal
        """
        gamma      = self.cfg.gamma
        gae_lambda = self.cfg.gae_lambda

        last_gae = np.zeros(self.N, dtype=np.float32)

        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            # TD error δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            # A_t = δ_t + γλ * (1 - done) * A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def reset(self):
        self.ptr  = 0
        self.full = False

    # ─────────────────────────────────────────────────────────────────────────
    # Mini-batch iteration
    # ─────────────────────────────────────────────────────────────────────────

    def get_mini_batches(
        self,
        batch_size: int,
        normalize_advantages: bool = True,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Flatten [T, N] → [T*N] and yield random mini-batches of size batch_size.
        Normalizes advantages within each full rollout (not per mini-batch).
        """
        assert self.full, "Buffer must be full before iterating mini-batches."

        total = self.T * self.N

        # Flatten
        obs_flat       = self.obs.reshape(total, self.obs_dim)
        actions_flat   = self.actions.reshape(total, self.act_dim)
        log_probs_flat = self.log_probs.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        returns_flat   = self.returns.reshape(total)
        values_flat    = self.values.reshape(total)

        # Normalize advantages over the whole rollout
        if normalize_advantages:
            adv_mean = advantages_flat.mean()
            adv_std  = advantages_flat.std() + 1e-8
            advantages_flat = (advantages_flat - adv_mean) / adv_std

        # Shuffle & yield
        indices = np.random.permutation(total)
        start = 0
        while start < total:
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield {
                "obs":        torch.tensor(obs_flat[idx],       device=self.device),
                "actions":    torch.tensor(actions_flat[idx],   device=self.device),
                "log_probs":  torch.tensor(log_probs_flat[idx], device=self.device),
                "advantages": torch.tensor(advantages_flat[idx],device=self.device),
                "returns":    torch.tensor(returns_flat[idx],   device=self.device),
                "values":     torch.tensor(values_flat[idx],    device=self.device),
            }
            start = end

    def num_samples(self) -> int:
        return self.T * self.N
