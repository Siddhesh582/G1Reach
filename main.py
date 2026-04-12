"""
main.py — Entry point for VisionGuidedPolicy PPO training (MuJoCo)

Usage (on Hulk PC, inside G1vision conda env):
  # Smoke test — CPU, 2000 steps, confirms everything imports correctly
  python main.py --test

  # Full training on GPU 1
  CUDA_VISIBLE_DEVICES=1 python main.py --train

  # Resume from checkpoint
  CUDA_VISIBLE_DEVICES=1 python main.py --train --resume checkpoints/ppo_g1_reach_v2_step500000.pt

  # Evaluate a saved policy
  CUDA_VISIBLE_DEVICES=1 python main.py --eval --checkpoint checkpoints/ppo_g1_reach_v2_final.pt

  # Watch TensorBoard (separate terminal)
  tensorboard --logdir logs/
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import G1Config
from training.ppo_trainer import PPOTrainer
from env.g1_reach_env import G1ReachEnv
from policy.actor_critic import ActorCritic


def parse_args():
    p = argparse.ArgumentParser(description="VisionGuidedPolicy — G1 PPO (MuJoCo)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Run full training")
    mode.add_argument("--test",  action="store_true", help="Smoke test on CPU (~30s)")
    mode.add_argument("--eval",  action="store_true", help="Evaluate saved checkpoint")

    p.add_argument("--resume",      type=str,   default=None, help="Checkpoint to resume from")
    p.add_argument("--checkpoint",  type=str,   default=None, help="Checkpoint for --eval")
    p.add_argument("--num_envs",    type=int,   default=None, help="Override num_envs")
    p.add_argument("--total_steps", type=int,   default=None, help="Override total_timesteps")
    p.add_argument("--device",      type=str,   default=None, help="Override device (e.g. cuda:1)")
    p.add_argument("--run_name",    type=str,   default=None, help="Override run name")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(cfg: G1Config):
    print("\n[SmokeTest] Starting...")
    cfg.device   = "cpu"
    cfg.num_envs = 2

    # 1. Environment
    print("[SmokeTest] Instantiating G1ReachEnv...")
    env = G1ReachEnv(cfg=cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (cfg.stacked_obs_dim,), \
        f"Obs shape mismatch: got {obs.shape}, expected ({cfg.stacked_obs_dim},)"
    assert env.action_space.shape == (cfg.action_dim,), \
        f"Action shape mismatch: {env.action_space.shape}"
    print(f"  obs shape:    {obs.shape}  ✓")
    print(f"  action shape: {env.action_space.shape}  ✓")

    # 2. Policy
    print("[SmokeTest] Instantiating ActorCritic...")
    policy = ActorCritic(cfg)
    print(f"  parameters: {policy.n_parameters:,}  ✓")

    # 3. Random rollout
    print("[SmokeTest] Running 200 random steps...")
    total_reward = 0.0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"  total random reward (200 steps): {total_reward:.3f}  ✓")

    # 4. Policy forward pass
    print("[SmokeTest] Policy forward pass...")
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_t, log_prob, entropy, value = policy.get_action_and_value(obs_t)
    print(f"  action:   {action_t.numpy().flatten()[:4].round(3)}...")
    print(f"  log_prob: {log_prob.item():.4f}")
    print(f"  entropy:  {entropy.item():.4f}")
    print(f"  value:    {value.item():.4f}")

    # 5. Check obs history stacking
    print(f"[SmokeTest] Obs history: {cfg.history_length} frames × {cfg.obs_dim} = {cfg.stacked_obs_dim}  ✓")

    env.close()
    print("\n[SmokeTest] ✓ All checks passed — ready to train.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(cfg: G1Config, checkpoint_path: str):
    if checkpoint_path is None:
        raise ValueError("--eval requires --checkpoint <path>")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(cfg).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"[Eval] Loaded checkpoint: {checkpoint_path}")

    env = G1ReachEnv(cfg=cfg)
    rewards, successes, dists = [], [], []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward, done = 0.0, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        successes.append(float(info.get("success", False)))
        dists.append(info.get("dist", float("inf")))
        print(f"  [Ep {ep+1:2d}] reward={ep_reward:+.2f}  "
              f"success={info.get('success')}  dist={info.get('dist', 0):.4f}m")

    env.close()
    print(f"\n[Eval] mean_reward={np.mean(rewards):+.3f}  "
          f"success={np.mean(successes):.2%}  mean_dist={np.mean(dists):.4f}m")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = G1Config()

    # CLI overrides
    if args.num_envs:    cfg.num_envs        = args.num_envs
    if args.total_steps: cfg.total_timesteps = args.total_steps
    if args.device:      cfg.device          = args.device
    if args.run_name:    cfg.run_name        = args.run_name

    if args.test:
        run_smoke_test(cfg)
        return

    if args.eval:
        run_eval(cfg, args.checkpoint)
        return

    if args.train:
        # GPU check
        if "cuda" in cfg.device and not torch.cuda.is_available():
            print("[Warning] CUDA not available — falling back to CPU")
            cfg.device = "cpu"
        else:
            if "cuda" in cfg.device:
                idx = int(cfg.device.split(":")[-1]) if ":" in cfg.device else 0
                print(f"[main] GPU: {torch.cuda.get_device_name(idx)}")

        trainer = PPOTrainer(cfg)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        try:
            trainer.train()
        finally:
            trainer.close()


if __name__ == "__main__":
    main()