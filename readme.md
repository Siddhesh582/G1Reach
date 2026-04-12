# VisionGuidedPolicy
### Vision-Guided RL Manipulation on Unitree G1
**Course:** CS 5180 — Reinforcement Learning  
**Author:** Siddhesh Santosh Shingate  

---

## Overview

This project trains a PPO-based reinforcement learning policy that moves the Unitree G1 robot's right-hand end-effector to arbitrary 3D target positions. In simulation, targets are sampled randomly within the arm's reachable workspace. On real hardware, targets are detected live from an Intel RealSense D435i camera using AprilTag fiducial markers.

---

## Project Structure

```
VisionGuidedPolicy/
├── config.py                   # All hyperparameters — edit here only
├── apriltag_detector.py        # Real hardware: AprilTag pose detection (D435i)
│
├── env/
│   └── g1_reach_env.py         # Isaac Lab manager-based environment
│
├── policy/
│   └── actor_critic.py         # PPO Actor-Critic MLP networks
│
├── training/
│   ├── ppo_trainer.py          # PPO update loop (clipped objective + GAE)
│   └── rollout_buffer.py       # Rollout storage and advantage estimation
│
├── deploy/
│   └── deploy.py               # Hardware deployment: detector → policy → G1
│
├── utils/
│   ├── transforms.py           # Coordinate frame transforms, rotation helpers
│   └── logger.py               # TensorBoard + CSV logging
│
├── main.py                     # Entry point for training
├── requirements.txt
├── checkpoints/                # Saved model weights (git-ignored)
└── logs/                       # Training curves (git-ignored)
```

---

## Setup

### 1. Prerequisites

- Ubuntu 22.04
- Python 3.10
- NVIDIA GPU with CUDA 11.8+
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) installed and sourced

### 2. Clone and install dependencies

```bash
cd ~/RL_CS5180/VisionGuidedPolicy
pip install -r requirements.txt
```

### 3. Verify Isaac Lab can see the G1 asset

```bash
python -c "from isaaclab_assets.robots.unitree import UNITREE_G1_CFG; print('G1 asset OK')"
```

---

## Training

```bash
# From VisionGuidedPolicy/
python main.py
```

**Common overrides:**

```bash
# Headless (no GUI) — for servers / cluster
python main.py --headless

# Change number of parallel environments
python main.py --num_envs 512

# Resume from checkpoint
python main.py --checkpoint checkpoints/iter_1000.pt
```

Training logs are written to `logs/`. Monitor with TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## Observation Space (32 dims)

| Slice | Dims | Description |
|-------|------|-------------|
| `obs[0:3]` | 3 | Target position (x, y, z) in robot base frame |
| `obs[3:16]` | 13 | Arm joint positions (radians) |
| `obs[16:29]` | 13 | Arm joint velocities (rad/s, clipped) |
| `obs[29:32]` | 3 | End-effector position (x, y, z) in base frame |

---

## Action Space (13 dims)

Joint position offsets for all 13 upper-body joints, scaled by `action_scale = 0.25 rad`.  
Actions are clipped to `[-1, 1]` before scaling.

---

## Reward Function

```
r = - alpha  × distance(ee, target)          # distance penalty
    + gamma  × success_bonus                  # sparse: +10 if within 5 cm
    + prog   × progress                       # dense: reward for getting closer
    - delta  × energy_cost                    # joint velocity penalty
    - lambda × joint_limit_penalty            # penalty near joint limits
```

Default weights: `alpha=1.0`, `gamma=10.0`, `progress=2.0`, `delta=0.01`.  
Obstacle avoidance penalty (`beta`) is set to 0 in Phase 1.

---

## PPO Configuration

| Parameter | Value |
|-----------|-------|
| Network | 256 → 128 → 64 (separate actor / critic) |
| Activation | ELU |
| Learning rate | 3e-4 (linear decay) |
| Clip ε | 0.2 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| Parallel envs | 256 |
| Rollout steps | 24 per env |
| Batch size | 256 × 24 = 6144 |
| Mini-batch size | 6144 / 4 = 1536 |
| Max iterations | 3000 |

---

## Hardware Deployment (after sim training)

Requires Intel RealSense D435i mounted on G1 and the Unitree SDK.

```bash
# 1. Calibrate camera mount — measure translation + angles from base origin
# 2. Update T_cam_to_base in deploy/deploy.py
# 3. Print a tag36h11 ID=0 tag at exactly tag_size_m (default 10 cm)
# 4. Run:
python deploy/deploy.py --checkpoint checkpoints/best.pt
```

---

## Experiments

| # | Description | Key Metric |
|---|-------------|------------|
| 1 | Basic reaching — sim | Success rate (target: >85%) |
| 2 | PPO vs P-controller baseline | Success rate, trajectory smoothness |
| 3 | Reward ablation | Contribution of each reward term |
| 4 | Reaching with static obstacles | Success + collision rate (target: >80%, <5%) |

---

## Dependencies

See `requirements.txt`. Key packages:

- `torch` — PPO implementation
- `isaaclab` / `isaaclab_assets` — simulation + G1 robot model
- `pyrealsense2` — RealSense D435i driver *(deploy only)*
- `opencv-contrib-python` — AprilTag detection *(deploy only)*

---

## References

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
2. Rudin et al. (2022). *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2022
3. NVIDIA Isaac Lab Documentation
4. Unitree Robotics G1 SDK
