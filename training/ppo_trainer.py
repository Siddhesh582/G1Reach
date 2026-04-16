"""
training/ppo_trainer.py — PPO trainer aligned with unitree_rl_lab rsl_rl_ppo_cfg.py

Key changes from v1:
  - Adaptive KL-based learning rate schedule (schedule="adaptive", desired_kl=0.01)
    Matches rsl_rl OnPolicyRunner behaviour exactly
  - value_coef=1.0, entropy_coef=0.01, max_grad_norm=1.0 (from rsl_rl_ppo_cfg)
  - n_epochs=5, batch_size derived from num_mini_batches=4
  - Separate actor/critic networks (no shared backbone)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
import math as _math

from config import G1Config
from policy.actor_critic import ActorCritic
from training.rollout_buffer import RolloutBuffer
from utils.logger import TrainingLogger
from env.g1_reach_env import G1ReachEnv


class PPOTrainer:

    def __init__(self, cfg: G1Config):
        self.cfg    = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"[PPOTrainer] Device: {self.device}")

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if "cuda" in str(self.device):
            torch.cuda.manual_seed_all(cfg.seed)

        self.envs     = self._make_envs()
        self.eval_env = G1ReachEnv(cfg=cfg)
        self.policy   = ActorCritic(cfg).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
        print(f"[PPOTrainer] Parameters: {self.policy.n_parameters:,}")

        self.buffer = RolloutBuffer(cfg, self.device)

        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.logger = TrainingLogger(cfg.log_dir, cfg.run_name)

        self.global_step = 0
        self.rollout_num = 0
        self.current_lr  = cfg.learning_rate
        self.start_time  = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        steps_per_rollout = cfg.rollout_steps * cfg.num_envs

        obs, _ = self.envs.reset(seed=cfg.seed)
        obs    = torch.tensor(obs, dtype=torch.float32, device=self.device)
        dones  = torch.zeros(cfg.num_envs, device=self.device)

        print(f"\n[PPOTrainer] Training | total_steps={cfg.total_timesteps:,} | "
              f"envs={cfg.num_envs} | rollout={cfg.rollout_steps} | "
              f"lr_schedule={cfg.lr_schedule}")

        while self.global_step < cfg.total_timesteps:

            # ── Collect rollout ──────────────────────────────────────────────
            ep_rewards, ep_lengths, ep_successes = [], [], []
            buf_rewards = np.zeros(cfg.num_envs, dtype=np.float32)
            buf_lengths = np.zeros(cfg.num_envs, dtype=np.int32)
            # Debug: track distances seen during rollout to check arm is moving
            rollout_dists = []

            self.buffer.reset()
            self.policy.eval()

            for _ in range(cfg.rollout_steps):
                with torch.no_grad():
                    action_raw, log_prob, _, value = self.policy.get_action_and_value(obs)
                    action_env = action_raw.clamp(-1.0, 1.0)

                cpu_act_raw = action_raw.cpu().numpy()   # store this in buffer
                cpu_act_env = action_env.cpu().numpy()   # send this to env

                next_obs, reward, terminated, truncated, info = self.envs.step(cpu_act_env)

                done = np.logical_or(terminated, truncated)

                self.buffer.add(
                    obs=obs.cpu().numpy(),
                    action=cpu_act_raw,
                    reward=reward.astype(np.float32),
                    value=value.cpu().numpy(),
                    log_prob=log_prob.cpu().numpy(),
                    done=done.astype(np.float32),
                )

                buf_rewards += reward
                buf_lengths += 1
                for i in range(cfg.num_envs):
                    if done[i]:
                        ep_rewards.append(buf_rewards[i])
                        ep_lengths.append(buf_lengths[i])
                        if "final_info" in info and info["final_info"][i] is not None:
                            ep_successes.append(
                                float(info["final_info"][i].get("success", False))
                            )
                        buf_rewards[i] = 0
                        buf_lengths[i] = 0

                obs   = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                dones = torch.tensor(done,     dtype=torch.float32, device=self.device)
                self.global_step += cfg.num_envs

                # Collect dist for debugging (sample env 0 only)
                if "dist" in info:
                    d = info["dist"]
                    rollout_dists.append(float(d[0]) if hasattr(d, '__len__') else float(d))

            # ── GAE ──────────────────────────────────────────────────────────
            with torch.no_grad():
                last_values = self.policy.get_value(obs).cpu().numpy()
            self.buffer.compute_returns(last_values, dones.cpu().numpy())

            # ── PPO update ───────────────────────────────────────────────────
            update_stats = self._ppo_update()
            self.rollout_num += 1

            # ── Adaptive LR (from rsl_rl schedule="adaptive") ────────────────
            if cfg.lr_schedule == "adaptive":
                kl = update_stats["update/approx_kl"]
                if kl > cfg.desired_kl * 2.0:
                    self.current_lr = max(self.current_lr / 1.5, cfg.lr_min)
                elif kl < cfg.desired_kl / 2.0 and kl > 0.0:
                    self.current_lr = min(self.current_lr * 1.5, cfg.learning_rate)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.current_lr
            elif cfg.lr_schedule == "linear":
                frac = 1.0 - self.global_step / cfg.total_timesteps
                self.current_lr = max(cfg.learning_rate * frac, cfg.lr_min)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.current_lr

            # ── log_std annealing ─────────────────────────────────────────────
            # Decay exploration noise linearly: std=1.0 (start) → std=0.135 (end)
            # Decouples exploration from entropy loss — prevents log_std freezing.
            progress = self.global_step / cfg.total_timesteps
            log_std_now = cfg.log_std_init + progress * (cfg.log_std_final - cfg.log_std_init)
            self.policy.log_std_value = float(log_std_now)


            # ── Logging ──────────────────────────────────────────────────────
            if self.rollout_num % cfg.log_interval == 0:
                elapsed = time.time() - self.start_time
                fps     = self.global_step / elapsed
                stats = {
                    "train/mean_reward":    np.mean(ep_rewards)    if ep_rewards    else 0.0,
                    "train/mean_ep_length": np.mean(ep_lengths)    if ep_lengths    else 0.0,
                    "train/success_rate":   np.mean(ep_successes)  if ep_successes  else 0.0,
                    "train/dist_mean":      np.mean(rollout_dists) if rollout_dists else 0.0,  # ADD THIS
                    "train/fps":            fps,
                    "train/learning_rate":  self.current_lr,
                    **update_stats,
                }

                self.logger.log(stats, step=self.global_step)
                print(
                    f"[{self.global_step:>8,}] "
                    f"rew={stats['train/mean_reward']:+.3f}  "
                    f"succ={stats['train/success_rate']:.2%}  "
                    f"kl={update_stats['update/approx_kl']:.4f}  "
                    f"lr={self.current_lr:.2e}  "
                    f"fps={fps:.0f}"
                )
                # ── Debug diagnostics ─────────────────────────────────────────
                import math as _math
                _log_std = self.policy.log_std_value
                _std     = _math.exp(_log_std)
                print(
                    f"  [std]     current={_std:.4f}  "
                    f"log_std={_log_std:.4f}  "
                    f"progress={self.global_step/cfg.total_timesteps:.3f}"
                )
                print(
                    f"  [loss]    pol={update_stats['update/policy_loss']:+.4f}  "
                    f"val={update_stats['update/value_loss']:.4f}  "
                    f"ent={update_stats['update/entropy']:.4f}  "
                    f"clip={update_stats['update/clip_frac']:.3f}"
                )
                if rollout_dists:
                    print(
                        f"  [dist]    mean={np.mean(rollout_dists):.4f}m  "
                        f"min={np.min(rollout_dists):.4f}m  "
                        f"max={np.max(rollout_dists):.4f}m  "
                        f"(env0, {len(rollout_dists)} steps)"
                    )

            # ── Evaluation ───────────────────────────────────────────────────
            if self.rollout_num % cfg.eval_interval == 0:
                ev = self._evaluate()
                self.logger.log({
                    "eval/mean_reward":  ev["mean_reward"],
                    "eval/success_rate": ev["success_rate"],
                    "eval/mean_dist":    ev["mean_dist"],
                }, step=self.global_step)
                print(f"  [EVAL] rew={ev['mean_reward']:+.3f}  "
                      f"succ={ev['success_rate']:.2%}  dist={ev['mean_dist']:.4f}m")

            # ── Checkpoint ───────────────────────────────────────────────────
            if self.rollout_num % cfg.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint(tag="final")
        self.logger.close()
        print(f"\n[PPOTrainer] Done — {self.global_step:,} steps")

    # ─────────────────────────────────────────────────────────────────────────
    # PPO update  (coefficients from rsl_rl_ppo_cfg.py)
    # ─────────────────────────────────────────────────────────────────────────

    def _ppo_update(self):
        cfg = self.cfg
        self.policy.train()

        pol_losses, val_losses, entropies, clip_fracs, kls = [], [], [], [], []

        for _ in range(cfg.n_epochs):  # num_learning_epochs=5
            for batch in self.buffer.get_mini_batches(cfg.batch_size):
                obs        = batch["obs"]
                actions    = batch["actions"]
                old_lp     = batch["log_probs"]
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns    = batch["returns"]
                old_values = batch["values"]

                new_lp, entropy, new_values = self.policy.evaluate_actions(obs, actions)

                log_ratio = new_lp - old_lp
                ratio     = log_ratio.exp()

                # Clipped policy loss
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss (use_clipped_value_loss=True)
                v_clipped   = old_values + (new_values - old_values).clamp(-cfg.clip_eps, cfg.clip_eps)
                v_loss      = torch.max(
                    (new_values - returns).pow(2),
                    (v_clipped  - returns).pow(2)
                ).mean()
                value_loss  = 0.5 * v_loss 

                # Entropy
                entropy_loss = -entropy.mean()

                # Total loss — value_loss_coef=1.0, entropy_coef=0.01
                loss = (
                    policy_loss
                    + cfg.value_coef   * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                if grad_norm > 1.0:
                    print(f"  [GRAD] norm={grad_norm:.2f} — clipped")
                self.optimizer.step()


                with torch.no_grad():
                    has_nan = any(p.isnan().any() for p in self.policy.parameters())
                    if has_nan:
                        print("  [NaN] detected in policy weights — stopping update")
                        break

                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac  = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()

                    # Detect KL spike — print per-joint std at the moment of explosion
                    if approx_kl > 0.1:
                        _lsv = self.policy.log_std_value
                        print(
                            f"  [KL-SPIKE] kl={approx_kl:.4f}  "
                            f"ratio min={ratio.min().item():.4f}  "
                            f"max={ratio.max().item():.4f}  "
                            f"log_std={_lsv:.4f}  std={_math.exp(_lsv):.4f}"
                        )

                pol_losses.append(policy_loss.item())
                val_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                clip_fracs.append(clip_frac)
                kls.append(approx_kl)

        return {
            "update/policy_loss": np.mean(pol_losses),
            "update/value_loss":  np.mean(val_losses),
            "update/entropy":     np.mean(entropies),
            "update/clip_frac":   np.mean(clip_fracs),
            "update/approx_kl":   np.mean(kls),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self):
        self.policy.eval()
        rewards, successes, dists = [], [], []
        for _ in range(self.cfg.eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_rew, done = 0.0, False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = self.policy.get_action_and_value(obs_t, deterministic=True)
                obs, rew, terminated, truncated, info = self.eval_env.step(
                    action.cpu().numpy()[0]
                )
                ep_rew += rew
                done    = terminated or truncated
            rewards.append(ep_rew)
            successes.append(float(info.get("success", False)))
            dists.append(info.get("dist", float("inf")))
        return {
            "mean_reward":  float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_dist":    float(np.mean(dists)),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag=""):
        tag_str = f"_{tag}" if tag else f"_step{self.global_step}"
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"{self.cfg.run_name}{tag_str}.pt"
        )
        torch.save({
            "global_step":           self.global_step,
            "rollout_num":           self.rollout_num,
            "current_lr":            self.current_lr,
            "policy_state_dict":     self.policy.state_dict(),
            "optimizer_state_dict":  self.optimizer.state_dict(),
        }, path)
        print(f"  [Checkpoint] → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.rollout_num = ckpt.get("rollout_num", 0)
        self.current_lr  = ckpt.get("current_lr", self.cfg.learning_rate)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr
        print(f"[PPOTrainer] Loaded {path} (step {self.global_step:,})")

    # ─────────────────────────────────────────────────────────────────────────
    # Env factory
    # ─────────────────────────────────────────────────────────────────────────

    def _make_envs(self):
        cfg = self.cfg
        def make_env(rank):
            def _init():
                env = G1ReachEnv(cfg=cfg)
                env.reset(seed=cfg.seed + rank)
                return env
            return _init
        return gym.vector.SyncVectorEnv([make_env(i) for i in range(cfg.num_envs)])

    def close(self):
        self.envs.close()
        self.eval_env.close()
        self.logger.close()