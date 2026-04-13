"""
policy/actor_critic.py — Actor-Critic network aligned with unitree_rl_lab rsl_rl_ppo_cfg.py

From rsl_rl_ppo_cfg.py:
    actor_hidden_dims  = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation         = "elu"
    init_noise_std     = 1.0  → log_std_init = 0.0

Key difference from v1: actor and critic have SEPARATE backbones (not shared).
This matches rsl_rl's ActorCritic implementation and gives the critic access to
privileged information if needed later.

Input:  stacked_obs_dim = 225  (45-dim × history_length=5)
Output: action_dim      = 14   (14 upper-body joints)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional

from config import G1Config


def make_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    activation: str = "elu",
    ortho_init: bool = True,
    output_scale: float = 0.01,
) -> nn.Sequential:
    act_fn = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[activation]
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        lin = nn.Linear(in_dim, h)
        if ortho_init:
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
            nn.init.zeros_(lin.bias)
        layers += [lin, act_fn()]
        in_dim = h
    out = nn.Linear(in_dim, output_dim)
    if ortho_init:
        nn.init.orthogonal_(out.weight, gain=output_scale)
        nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Separate actor and critic MLPs, aligned with rsl_rl ActorCritic.

    Actor:  obs → [512, 256, 128] → ELU → mean (action_dim=14)
    Critic: obs → [512, 256, 128] → ELU → value (1)
    log_std: learned parameter, initialised to 0.0 (std=1.0)
    """

    def __init__(self, cfg: G1Config = None):
        super().__init__()
        self.cfg = cfg or G1Config()

        in_dim     = self.cfg.stacked_obs_dim    # 225  (was 210)
        act_dim    = self.cfg.action_dim          # 14   (was 13)
        act_hidden = self.cfg.actor_hidden_dims   # (512, 256, 128)
        crt_hidden = self.cfg.critic_hidden_dims  # (512, 256, 128)
        activation = self.cfg.activation          # "elu"
        ortho      = self.cfg.ortho_init

        # ── Actor ─────────────────────────────────────────────────────────────
        self.actor = make_mlp(
            in_dim, act_hidden, act_dim,
            activation=activation, ortho_init=ortho, output_scale=0.01
        )

        # Learnable log_std — init_noise_std=1.0 → log_std_init=0.0
        self.log_std = nn.Parameter(
            torch.full((act_dim,), self.cfg.log_std_init)
        )

        # ── Critic ────────────────────────────────────────────────────────────
        self.critic = make_mlp(
            in_dim, crt_hidden, 1,
            activation=activation, ortho_init=ortho, output_scale=1.0
        )

    # ──────────────────────────────────────────────────────────────────────────

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean  = self.actor(obs)
        std   = self.log_std.exp().expand_as(mean)
        dist  = Normal(mean, std)

        if action is None:
            action = mean if deterministic else dist.rsample()
        action = action.clamp(-1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean  = self.actor(obs)
        std   = self.log_std.exp().expand_as(mean)
        dist  = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
