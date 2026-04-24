"""
main.py - Entry point for VisionGuidedPolicy PPO training (MuJoCo)

Usage:
  # Smoke test
  python main.py --test

  # Full training on GPU 1
  CUDA_VISIBLE_DEVICES=1 python main.py --train

  # Resume from checkpoint
  CUDA_VISIBLE_DEVICES=1 python main.py --train --resume checkpoints/ppo_g1_reach_v2_step500000.pt

  # Evaluate (numbers only, no render)
  CUDA_VISIBLE_DEVICES=1 python main.py --eval --checkpoint checkpoints/ppo_g1_reach_v2_final.pt

  # Watch trained policy in MuJoCo viewer at quarter speed
  CUDA_VISIBLE_DEVICES=1 python main.py --watch --checkpoint checkpoints/ppo_g1_reach_v5_final.pt --speed 0.25

  # Record video - runs forever until Ctrl+C (clean 1080p, no overlays)
  CUDA_VISIBLE_DEVICES=1 python main.py --record --checkpoint checkpoints/ppo_g1_reach_v5_final.pt --record-speed 0.1

  # Record a specific number of episodes
  CUDA_VISIBLE_DEVICES=1 python main.py --record --checkpoint checkpoints/ppo_g1_reach_v5_final.pt --episodes 20 --record-speed 0.1

  # Lower resolution if 1080p is too slow
  CUDA_VISIBLE_DEVICES=1 python main.py --record --checkpoint checkpoints/ppo_g1_reach_v5_final.pt --record-speed 0.1 --width 1280 --height 720

  # TensorBoard
  tensorboard --logdir logs/
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import G1Config
from training.ppo_trainer import PPOTrainer
from env.g1_reach_env import G1ReachEnv
from policy.actor_critic import ActorCritic


def parse_args():
    p = argparse.ArgumentParser(description="VisionGuidedPolicy - G1 PPO (MuJoCo)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",  action="store_true", help="Run full training")
    mode.add_argument("--test",   action="store_true", help="Smoke test on CPU (~30s)")
    mode.add_argument("--eval",   action="store_true", help="Evaluate saved checkpoint (no render)")
    mode.add_argument("--watch",  action="store_true", help="Watch policy in MuJoCo viewer")
    mode.add_argument("--record", action="store_true", help="Record episode videos to disk")

    p.add_argument("--resume",        type=str,   default=None)
    p.add_argument("--checkpoint",    type=str,   default=None)
    p.add_argument("--num_envs",      type=int,   default=None)
    p.add_argument("--total_steps",   type=int,   default=None)
    p.add_argument("--device",        type=str,   default=None)
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--episodes",      type=int,   default=0,
                   help="Episodes to run. 0 = run forever (Ctrl+C to stop). Works for --watch and --record.")
    p.add_argument("--speed",         type=float, default=1.0,
                   help="Playback speed for --watch (0.25=quarter, 0.5=half, 1.0=realtime)")
    p.add_argument("--record-speed",  type=float, default=0.1,
                   help="Speed factor for recorded video (0.1=10pct realtime = 10x slow motion)")
    p.add_argument("--video-dir",     type=str,   default="videos",
                   help="Output directory for recorded videos (default: videos/)")
    p.add_argument("--width",         type=int,   default=1920,
                   help="Recording width in pixels (default: 1920). Use 1280 for faster rendering.")
    p.add_argument("--height",        type=int,   default=1080,
                   help="Recording height in pixels (default: 1080). Use 720 for faster rendering.")
    return p.parse_args()


# Codec selection - headless-safe

def _pick_codec(video_dir: str, width: int, height: int, fps: int):
    """
    Return (fourcc, ext) for the best available codec on this machine.

    Priority:
      1. avc1 / H.264  - best quality/size, hardware-encoded on desktop Linux
      2. mp4v / MPEG-4 - universal software fallback, always available
      3. XVID           - last resort

    On headless servers the V4L2 H.264 encoder is absent so avc1 will fail
    the test write. We detect that here and skip it cleanly without spewing
    FFmpeg error logs (stderr is suppressed during the probe).
    """
    import cv2

    candidates = [
        ("avc1", "mp4"),
        ("mp4v", "mp4"),
        ("XVID", "avi"),
    ]

    # Suppress FFmpeg/OpenCV error output during codec probe
    devnull = open(os.devnull, "w")
    old_stderr_fd = os.dup(2)
    os.dup2(devnull.fileno(), 2)

    chosen_fourcc = None
    chosen_ext    = None
    try:
        for codec_str, ext in candidates:
            fourcc    = cv2.VideoWriter_fourcc(*codec_str)
            test_path = os.path.join(video_dir, f"_probe_{codec_str}.{ext}")
            writer    = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            ok        = writer.isOpened()
            writer.release()
            if os.path.exists(test_path):
                os.remove(test_path)
            if ok:
                chosen_fourcc = fourcc
                chosen_ext    = ext
                break
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        devnull.close()

    if chosen_fourcc is None:
        raise RuntimeError(
            "No working video codec found. "
            "Install opencv-contrib-python or ffmpeg."
        )

    label = {
        cv2.VideoWriter_fourcc(*"avc1"): "H.264 (avc1)",
        cv2.VideoWriter_fourcc(*"mp4v"): "MPEG-4 (mp4v)",
        cv2.VideoWriter_fourcc(*"XVID"): "XVID",
    }.get(chosen_fourcc, "unknown")
    print(f"[Record] Codec:       {label}")
    return chosen_fourcc, chosen_ext


# Smoke test

def run_smoke_test(cfg: G1Config):
    print("\n[SmokeTest] Starting...")
    cfg.device   = "cpu"
    cfg.num_envs = 2

    print("[SmokeTest] Instantiating G1ReachEnv...")
    env = G1ReachEnv(cfg=cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (cfg.stacked_obs_dim,), \
        f"Obs shape mismatch: got {obs.shape}, expected ({cfg.stacked_obs_dim},)"
    assert env.action_space.shape == (cfg.action_dim,)
    print(f"  obs shape:    {obs.shape}  ok")
    print(f"  action shape: {env.action_space.shape}  ok")

    print("[SmokeTest] Instantiating ActorCritic...")
    policy = ActorCritic(cfg)
    print(f"  parameters: {policy.n_parameters:,}  ok")

    print("[SmokeTest] Running 200 random steps...")
    total_reward = 0.0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"  total random reward (200 steps): {total_reward:.3f}  ok")

    print("[SmokeTest] Policy forward pass...")
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_t, log_prob, entropy, value = policy.get_action_and_value(obs_t)
    print(f"  action:   {action_t.numpy().flatten()[:4].round(3)}...")
    print(f"  log_prob: {log_prob.item():.4f}")
    print(f"  entropy:  {entropy.item():.4f}")
    print(f"  value:    {value.item():.4f}")
    print(f"[SmokeTest] Obs history: {cfg.history_length} frames x {cfg.obs_dim} = {cfg.stacked_obs_dim}  ok")

    env.close()
    print("\n[SmokeTest] All checks passed - ready to train.\n")


# Evaluation (numbers only)

def run_eval(cfg: G1Config, checkpoint_path: str):
    if checkpoint_path is None:
        raise ValueError("--eval requires --checkpoint <path>")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(cfg).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"[Eval] Loaded: {checkpoint_path}")

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


# Watch - MuJoCo interactive viewer

def run_watch(cfg: G1Config, checkpoint_path: str, max_episodes: int = 0,
              speed: float = 1.0):
    """
    Render the trained policy in the MuJoCo interactive viewer.

    Args:
        speed:        Playback speed (0.25=quarter, 1.0=realtime).
        max_episodes: 0 = run forever until window closed.

    Controls:
      Space     - pause / resume
      Scroll    - zoom
      Left drag - rotate camera
    """
    if checkpoint_path is None:
        raise ValueError("--watch requires --checkpoint <path>")

    import mujoco
    import mujoco.viewer

    device = torch.device("cpu")
    policy = ActorCritic(cfg).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"[Watch] Loaded: {checkpoint_path}")
    if speed < 1.0:
        print(f"[Watch] Speed: {speed}x  ({1.0/speed:.1f}x slower than realtime)")
    else:
        print(f"[Watch] Speed: {speed}x (realtime)")
    if max_episodes > 0:
        print(f"[Watch] Will run {max_episodes} episodes then exit.")
    else:
        print(f"[Watch] Running indefinitely - close the viewer window to stop.")

    env = G1ReachEnv(cfg=cfg)
    obs, _ = env.reset(seed=0)

    ep_count  = 0
    ep_reward = 0.0
    step      = 0
    target_step_time = 1.0 / (50.0 * speed)

    def policy_step():
        nonlocal obs, ep_reward, step, ep_count
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.numpy()[0])
        ep_reward += reward
        step      += 1
        done = terminated or truncated
        if done:
            ep_count += 1
            print(f"  [Ep {ep_count:3d}] reward={ep_reward:+.2f}  "
                  f"success={info.get('success')}  "
                  f"dist={info.get('dist', float('inf')):.4f}m  steps={step}")
            obs, _ = env.reset()
            ep_reward = 0.0
            step = 0
            if max_episodes > 0 and ep_count >= max_episodes:
                return False
        return True

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.distance  = 2.5
        v.cam.elevation = -20
        v.cam.azimuth   = 135
        while v.is_running():
            t0 = time.time()
            if not policy_step():
                break
            v.sync()
            time.sleep(max(0.0, target_step_time - (time.time() - t0)))

    env.close()
    print(f"\n[Watch] Done - {ep_count} episodes.")


# Record - save episode videos to disk using offscreen renderer

def run_record(cfg: G1Config, checkpoint_path: str, max_episodes: int = 0,
               record_speed: float = 0.1, video_dir: str = "videos",
               width: int = 1920, height: int = 1080):
    """
    Record episode videos using MuJoCo's offscreen renderer.
    No display required - runs fully headless.

    max_episodes=0 means run forever until Ctrl+C.
    Each episode is saved individually as ep001_success.mp4 or ep001_fail.mp4.
    A combined all_episodes.mp4 is written when recording ends.

    How speed works:
      output_fps = 50 * record_speed
      record_speed=0.1 --> output_fps=5 --> 10x slow motion
      record_speed=0.5 --> output_fps=25 --> 2x slow motion
      record_speed=1.0 --> output_fps=50 --> realtime

    Resolution is controlled by --width / --height (default 1920x1080).
    The MuJoCo model XML must declare an offscreen framebuffer at least as
    large via <visual><global offwidth="W" offheight="H"/></visual>.

    Codec is selected automatically: H.264 if available, mp4v otherwise.
    No overlays - clean render only.
    """
    if checkpoint_path is None:
        raise ValueError("--record requires --checkpoint <path>")

    try:
        import cv2
    except ImportError:
        raise ImportError("Install opencv: pip install opencv-python")

    import mujoco

    os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cpu")
    policy = ActorCritic(cfg).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    output_fps = max(1, int(50.0 * record_speed))

    # Codec probe - silent, headless-safe
    fourcc, ext = _pick_codec(video_dir, width, height, output_fps)

    print(f"[Record] Loaded:      {checkpoint_path}")
    print(f"[Record] Episodes:    {'inf (Ctrl+C to stop)' if max_episodes == 0 else max_episodes}")
    print(f"[Record] Speed:       {record_speed}x --> {1/record_speed:.0f}x slow motion")
    print(f"[Record] Resolution:  {width}x{height}")
    print(f"[Record] Output FPS:  {output_fps}")
    print(f"[Record] Output dir:  {video_dir}/")

    env = G1ReachEnv(cfg=cfg, render_mode="rgb_array")

    # MuJoCo offscreen renderer.
    # Requires the scene XML to declare:
    #   <visual><global offwidth="W" offheight="H"/></visual>
    # where W >= width and H >= height. See scene_g1_reach.xml.
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    all_frames   = []
    ep_rewards   = []
    ep_successes = []
    ep_dists     = []
    ep_count     = 0

    try:
        while True:
            ep_count += 1
            obs, _ = env.reset(seed=ep_count)
            ep_reward = 0.0
            done      = False
            frames    = []

            while not done:
                # Render current state - clean, no overlays
                renderer.update_scene(env.data)
                frame     = renderer.render()               # RGB (H, W, 3)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
                all_frames.append(frame_bgr)

                # Policy step
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(obs_t, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action.numpy()[0])
                ep_reward += reward
                done = terminated or truncated

            success = info.get("success", False)
            dist    = info.get("dist", float("inf"))
            ep_rewards.append(ep_reward)
            ep_successes.append(float(success))
            ep_dists.append(dist)

            # Save per-episode video
            tag     = "success" if success else "fail"
            ep_path = os.path.join(video_dir, f"ep{ep_count:03d}_{tag}.{ext}")
            writer  = cv2.VideoWriter(ep_path, fourcc, output_fps, (width, height))
            for f in frames:
                writer.write(f)
            writer.release()

            print(f"  [Ep {ep_count:3d}] reward={ep_reward:+.2f}  success={success}  "
                  f"dist={dist:.4f}m  frames={len(frames)}  saved to {ep_path}")

            if max_episodes > 0 and ep_count >= max_episodes:
                break

    except KeyboardInterrupt:
        print(f"\n[Record] Interrupted at episode {ep_count} - saving combined video...")

    # Save combined video of everything recorded so far
    if all_frames:
        combined_path = os.path.join(video_dir, f"all_episodes.{ext}")
        writer = cv2.VideoWriter(combined_path, fourcc, output_fps, (width, height))
        for f in all_frames:
            writer.write(f)
        writer.release()
        print(f"  Combined video saved to {combined_path}")

    renderer.close()
    env.close()

    if ep_rewards:
        print(f"\n[Record] Summary")
        print(f"  Episodes recorded:  {ep_count}")
        print(f"  Success rate:       {np.mean(ep_successes):.1%}")
        print(f"  Mean reward:        {np.mean(ep_rewards):+.3f}")
        print(f"  Mean dist:          {np.mean(ep_dists):.4f}m")
        print(f"  Output FPS:         {output_fps}  ({1/record_speed:.0f}x slow motion)")
        print(f"[Record] Done.")


# Main

def main():
    args = parse_args()
    cfg  = G1Config()

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

    if args.watch:
        run_watch(cfg, args.checkpoint,
                  max_episodes=args.episodes,
                  speed=args.speed)
        return

    if args.record:
        run_record(cfg, args.checkpoint,
                   max_episodes=args.episodes,
                   record_speed=args.record_speed,
                   video_dir=args.video_dir,
                   width=args.width,
                   height=args.height)
        return

    if args.train:
        if "cuda" in cfg.device and not torch.cuda.is_available():
            print("[Warning] CUDA not available - falling back to CPU")
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