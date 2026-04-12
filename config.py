"""
config.py — Centralized configuration for VisionGuidedPolicy (MuJoCo PPO)
Unitree G1 upper-body reaching task | CS 5180 RL Project

Hyperparameters aligned with unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py
Observation scales aligned with unitree_rl_lab/tasks/locomotion/mdp/observations.py
Reward weights adapted from unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py
"""

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class G1Config:
    # ─── Paths ───────────────────────────────────────────────────────────────
    project_root:   str = os.path.dirname(os.path.abspath(__file__))
    asset_dir:      str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    model_xml:      str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "g1_upper.xml")
    log_dir:        str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    # ─── Environment ─────────────────────────────────────────────────────────
    # Observation (42-dim, proprioceptive only, aligned with unitree_rl_lab):
    #   rel_target(3) + joint_pos_rel(13) + joint_vel_rel(13) + last_action(13)
    obs_dim:          int   = 42
    # With history stacking (history_length=5): 42 * 5 = 210
    history_length:   int   = 5
    stacked_obs_dim:  int   = 210   # obs_dim * history_length — actual network input

    action_dim:       int   = 13

    # Joint names controlled by the policy (must match MJCF actuator names)
    controlled_joints: Tuple[str, ...] = (
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    )

    # Right-hand end-effector site name in MJCF
    ee_site_name:     str   = "right_hand_site"

    # Target workspace: sphere radius (m) centred in front of robot
    target_radius:    float = 0.6
    target_center:    Tuple[float, float, float] = (0.5, 0.0, 0.8)

    # Success threshold (m)
    success_dist:     float = 0.05

    # Episode length
    max_episode_steps: int  = 500   # at 50Hz = 10s, matches unitree_rl_lab 20s / decimation

    # Physics: MuJoCo 200Hz, policy acts every 4 substeps → 50Hz
    # Matches unitree_rl_lab: sim.dt=0.005, decimation=4
    physics_hz:         int   = 200
    control_decimation: int   = 4

    # Action scale — from unitree_rl_lab ActionsCfg: scale=0.25
    # Policy output [-1,1] is multiplied by this before adding to default joint pos
    action_scale:       float = 0.25

    # Fall detection
    fall_height_threshold: float = 0.5

    # ─── Observation scales (from unitree_rl_lab ObservationsCfg) ────────────
    # Applied as multipliers before feeding to network
    obs_scale_joint_pos:  float = 1.0    # joint_pos_rel — no scaling
    obs_scale_joint_vel:  float = 0.05   # joint_vel_rel scale=0.05
    obs_scale_rel_target: float = 1.0    # our addition — no scaling needed

    # ─── Observation noise (uniform, from unitree_rl_lab) ────────────────────
    # Matches enable_corruption=True in PolicyCfg
    obs_noise_joint_pos:  float = 0.01   # ±0.01 uniform
    obs_noise_joint_vel:  float = 1.5    # ±1.5 uniform (applied before scale)
    obs_noise_rel_target: float = 0.005  # small noise on relative target

    # ─── Reward weights (adapted from velocity_env_cfg.py RewardsCfg) ────────
    # Task rewards
    rew_weight_dist:          float = -1.0    # dense: -||ee - target||
    rew_weight_success:       float = 5.0     # bonus on reaching goal
    rew_weight_alive:         float = 0.15    # from mdp.is_alive weight=0.15

    # Regularisation (directly from unitree_rl_lab weights)
    rew_weight_action_rate:   float = -0.05   # action_rate_l2 weight=-0.05
    rew_weight_joint_vel:     float = -0.001  # joint_vel_l2 weight=-0.001
    rew_weight_joint_acc:     float = -2.5e-7 # joint_acc_l2 weight=-2.5e-7
    rew_weight_energy:        float = -2e-5   # energy weight=-2e-5
    rew_weight_dof_limits:    float = -5.0    # dof_pos_limits weight=-5.0
    rew_weight_joint_dev:     float = -0.1    # joint_deviation_arms weight=-0.1
    rew_weight_fall:          float = -5.0    # large penalty on fall

    # ─── Domain Randomisation (from velocity_env_cfg.py EventCfg) ────────────
    # Friction randomisation (startup, per unitree physics_material event)
    dr_friction_range:        Tuple[float, float] = (0.3, 1.0)
    # Base mass perturbation (startup, per unitree add_base_mass event)
    dr_mass_range:            Tuple[float, float] = (-1.0, 3.0)
    # Joint reset velocity range (per unitree reset_robot_joints event)
    dr_joint_vel_range:       Tuple[float, float] = (-1.0, 1.0)
    # Motor strength perturbation (multiplicative)
    dr_motor_strength_range:  Tuple[float, float] = (0.9, 1.1)
    # PD gain perturbation (multiplicative)
    dr_kp_range:              Tuple[float, float] = (0.8, 1.2)

    # ─── PPO Hyperparameters (from rsl_rl_ppo_cfg.py) ────────────────────────
    num_envs:         int   = 8
    rollout_steps:    int   = 2048        # per env (unitree uses 24 with 4096 envs; we scale up)
    batch_size:       int   = 512         # num_mini_batches=4 → total/4
    n_epochs:         int   = 5           # num_learning_epochs=5
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.2         # clip_param=0.2
    clip_value_loss:  bool  = True        # use_clipped_value_loss=True
    entropy_coef:     float = 0.01        # entropy_coef=0.01
    value_coef:       float = 1.0         # value_loss_coef=1.0
    max_grad_norm:    float = 1.0         # max_grad_norm=1.0
    learning_rate:    float = 1e-3        # learning_rate=1.0e-3

    # Adaptive KL-based LR schedule (from rsl_rl_ppo_cfg: schedule="adaptive")
    lr_schedule:      str   = "adaptive"  # "adaptive" | "linear"
    desired_kl:       float = 0.01        # desired_kl=0.01
    lr_min:           float = 1e-5        # floor for adaptive schedule

    total_timesteps:  int   = 5_000_000

    # ─── Network Architecture (from rsl_rl_ppo_cfg.py) ───────────────────────
    # actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]
    actor_hidden_dims:  Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation:         str             = "elu"   # activation="elu"
    # init_noise_std=1.0 in unitree — we keep log_std_init=0.0 (std=1.0)
    log_std_init:       float           = 0.0
    ortho_init:         bool            = True

    # ─── Hardware ────────────────────────────────────────────────────────────
    device:           str   = "cuda:1"   # GPU 1 on Hulk PC
    seed:             int   = 42

    # ─── Logging ─────────────────────────────────────────────────────────────
    log_interval:     int   = 1
    save_interval:    int   = 100        # save_interval=100 from unitree
    eval_interval:    int   = 25
    eval_episodes:    int   = 10
    run_name:         str   = "ppo_g1_reach_v2"
