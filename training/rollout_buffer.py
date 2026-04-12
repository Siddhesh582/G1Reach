"""
training/rollout_buffer.py
--------------------------
Stores one rollout (num_steps × num_envs transitions) and computes
Generalized Advantage Estimation (GAE) returns for PPO updates.

Usage:
    buf = RolloutBuffer(num_steps=24, num_envs=256, obs_dim=32,
                        action_dim=13, device=device, cfg=G1Config().ppo)
    # collect:
    for t in range(num_steps):
        buf.add(obs, action, log_prob, reward, done, value)
    buf.compute_returns(last_value)
    # iterate mini-batches:
    for batch in buf.mini_batches():
        ...
    buf.reset()
"""

from __future__ import annotations

import torch
from typing import Iterator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PPOConfig


class RolloutBuffer:
    """
    Fixed-size circular buffer for one PPO rollout.

    Stores raw transitions then computes GAE advantages and
    discounted returns in a single vectorised pass over all envs.
    """

    def __init__(
        self,
        num_steps:  int,
        num_envs:   int,
        obs_dim:    int,
        action_dim: int,
        device:     torch.device | str,
        cfg:        PPOConfig,
    ):
        self.num_steps  = num_steps
        self.num_envs   = num_envs
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self.gamma      = cfg.gamma
        self.lam        = cfg.lam
        self.num_mini_batches = cfg.num_mini_batches

        self._ptr = 0       # current insertion index
        self._ready = False # True after compute_returns() is called

        self._alloc()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _alloc(self):
        """Pre-allocate all storage tensors once."""
        S, N = self.num_steps, self.num_envs
        d    = self.device

        self.obs      = torch.zeros((S, N, self.obs_dim),    device=d)
        self.actions  = torch.zeros((S, N, self.action_dim), device=d)
        self.log_probs= torch.zeros((S, N),                  device=d)
        self.rewards  = torch.zeros((S, N),                  device=d)
        self.dones    = torch.zeros((S, N),                  device=d)
        self.values   = torch.zeros((S, N),                  device=d)

        # Filled by compute_returns()
        self.advantages = torch.zeros((S, N), device=d)
        self.returns    = torch.zeros((S, N), device=d)

    def reset(self):
        """Clear the buffer for the next rollout. Does NOT reallocate."""
        self._ptr   = 0
        self._ready = False

    # ------------------------------------------------------------------
    # Data insertion
    # ------------------------------------------------------------------

    def add(
        self,
        obs:      torch.Tensor,   # (N, obs_dim)
        action:   torch.Tensor,   # (N, action_dim)
        log_prob: torch.Tensor,   # (N,)
        reward:   torch.Tensor,   # (N,)
        done:     torch.Tensor,   # (N,)  bool or float
        value:    torch.Tensor,   # (N,)
    ):
        """Insert one timestep of data from all envs."""
        assert self._ptr < self.num_steps, \
            "Buffer full — call compute_returns() then reset() first."

        t = self._ptr
        self.obs[t]       = obs.detach()
        self.actions[t]   = action.detach()
        self.log_probs[t] = log_prob.detach()
        self.rewards[t]   = reward.detach()
        self.dones[t]     = done.float().detach()
        self.values[t]    = value.detach()

        self._ptr += 1

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns(self, last_value: torch.Tensor):
        """
        Compute GAE advantages and discounted returns.

        Must be called after all num_steps have been added, passing
        V(s_{T+1}) — the critic's estimate of the value of the state
        AFTER the last collected step (bootstrapping for non-terminal envs).

        Args:
            last_value : (N,) — critic value of the state after last step
        """
        assert self._ptr == self.num_steps, \
            f"Buffer not full: {self._ptr}/{self.num_steps} steps added."

        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            # Bootstrap from last_value on the final step
            next_val   = last_value if t == self.num_steps - 1 \
                         else self.values[t + 1]
            next_done  = self.dones[t]                  # 1.0 if episode ended

            # TD error: δ_t = r_t + γ V(s_{t+1}) (1 - done) - V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_val * (1.0 - next_done)
                - self.values[t]
            )

            # GAE: A_t = δ_t + γλ (1-done) A_{t+1}
            gae = delta + self.gamma * self.lam * (1.0 - next_done) * gae

            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

        # Normalise advantages across the whole rollout batch
        # (across all steps and envs) for stable PPO updates
        adv_flat = self.advantages.reshape(-1)
        self.advantages = (
            (self.advantages - adv_flat.mean()) /
            (adv_flat.std() + 1e-8)
        )

        self._ready = True

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def mini_batches(
        self,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield shuffled mini-batches for PPO update epochs.

        Flattens the (num_steps, num_envs) dimensions into a single
        batch axis, then splits into num_mini_batches chunks.

        Yields dicts with keys:
            obs, actions, log_probs_old, advantages, returns
        """
        assert self._ready, "Call compute_returns() before iterating mini-batches."

        total = self.num_steps * self.num_envs
        mb_sz = total // self.num_mini_batches

        # Flatten step × env into a single batch dimension
        obs_f       = self.obs.reshape(total, self.obs_dim)
        act_f       = self.actions.reshape(total, self.action_dim)
        lp_f        = self.log_probs.reshape(total)
        adv_f       = self.advantages.reshape(total)
        ret_f       = self.returns.reshape(total)

        # Random permutation for each call (i.e. each PPO epoch)
        idx = torch.randperm(total, device=self.device)

        for start in range(0, total, mb_sz):
            mb_idx = idx[start: start + mb_sz]
            yield {
                "obs":           obs_f[mb_idx],
                "actions":       act_f[mb_idx],
                "log_probs_old": lp_f[mb_idx],
                "advantages":    adv_f[mb_idx],
                "returns":       ret_f[mb_idx],
            }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def mean_reward(self) -> float:
        return self.rewards.mean().item()

    @property
    def mean_value(self) -> float:
        return self.values.mean().item()

    @property
    def explained_variance(self) -> float:
        """
        Fraction of return variance explained by the value function.
        Close to 1.0 = critic is well-calibrated.
        Close to 0.0 = critic is not learning.
        Negative     = critic is worse than a constant baseline.
        """
        ret_f = self.returns.reshape(-1)
        val_f = self.values.reshape(-1)
        var_ret = ret_f.var()
        if var_ret < 1e-8:
            return float("nan")
        return (1.0 - (ret_f - val_f).var() / var_ret).item()