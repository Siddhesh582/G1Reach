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
    asset_dir:      str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "g1")
    model_xml:      str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "g1", "scene_g1_reach.xml")
    log_dir:        str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    # ─── Environment ─────────────────────────────────────────────────────────
    # Observation (45-dim, proprioceptive only):
    #   rel_target(3) + joint_pos_rel(14) + joint_vel_rel(14) + last_action(14)
    obs_dim:          int   = 45
    # With history stacking (history_length=5): 45 * 5 = 225
    history_length:   int   = 5
    stacked_obs_dim:  int   = 225  # obs_dim * history_length — actual network input

    action_dim:       int   = 14

    # Joint names controlled by the policy.
    # Must match joint names in g1_29dof.xml exactly.
    # Order matches CONTROLLED_JOINTS in g1_reach_env.py.
    controlled_joints: Tuple[str, ...] = (
        # Waist (3)
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        # Right arm (7) — full arm including wrist, this is the reaching limb
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        # Left arm (4) — shoulder + elbow for balance; wrists excluded
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    )

    # Right-hand end-effector site name in g1_29dof.xml
    ee_site_name:     str   = "right_hand_ee"

    # Target workspace: sphere of this radius centred at target_center
    target_radius:    float = 0.15
    target_center:    Tuple[float, float, float] = (0.45, -0.15, 1.05)

    # Success threshold (m)
    success_dist:     float = 0.12

    # Episode length
    max_episode_steps: int  = 500   # at 50Hz = 10s

    # Physics: MuJoCo 200Hz, policy acts every 4 substeps → 50Hz
    # Matches unitree_rl_lab: sim.dt=0.005, decimation=4
    physics_hz:         int   = 200
    control_decimation: int   = 4

    # Action scale — from unitree_rl_lab ActionsCfg: scale=0.25
    # q_des = q_default + action * action_scale
    action_scale:       float = 0.4

    # Fall detection: pelvis z below this height (m) = fallen
    # G1 nominal pelvis height = 0.793 m; 0.5 m gives ~30 cm of drop tolerance
    fall_height_threshold: float = 0.5

    # ─── Observation scales (from unitree_rl_lab ObservationsCfg) ────────────
    obs_scale_joint_pos:  float = 1.0    # joint_pos_rel — no scaling
    obs_scale_joint_vel:  float = 0.05   # joint_vel_rel scale=0.05
    obs_scale_rel_target: float = 1.0    # relative target — no scaling

    # ─── Observation noise (uniform, from unitree_rl_lab) ────────────────────
    obs_noise_joint_pos:  float = 0.01   # ±0.01 uniform
    obs_noise_joint_vel:  float = 1.5    # ±1.5 uniform (applied before scale)
    obs_noise_rel_target: float = 0.005  # small noise on relative target

    # ─── Reward weights (adapted from velocity_env_cfg.py RewardsCfg) ────────
    rew_weight_dist:          float = -2.0    # dense: -||ee - target||
    rew_weight_success:       float = 10.0     # bonus on reaching goal
    rew_weight_alive:         float = 0.0    # from mdp.is_alive weight=0.15
    rew_weight_action_rate:   float = -0.001   # action_rate_l2 weight=-0.05
    rew_weight_joint_vel:     float = -0.0001  # joint_vel_l2 weight=-0.001
    rew_weight_joint_acc:     float = 0.0 # joint_acc_l2 weight=-2.5e-7
    rew_weight_energy:        float = 0.0   # energy weight=-2e-5
    rew_weight_dof_limits:    float = -0.5    # dof_pos_limits weight=-5.0
    rew_weight_joint_dev:     float = 0.0    # joint_deviation_arms weight=-0.1
    rew_weight_fall:          float = -10.0    # large penalty on fall

    # ─── Domain Randomisation (from velocity_env_cfg.py EventCfg) ────────────
    dr_friction_range:        Tuple[float, float] = (0.3, 1.0)
    dr_mass_range:            Tuple[float, float] = (-1.0, 3.0)
    dr_joint_vel_range:       Tuple[float, float] = (-1.0, 1.0)
    dr_motor_strength_range:  Tuple[float, float] = (0.9, 1.1)
    dr_kp_range:              Tuple[float, float] = (0.8, 1.2)

    # ─── PPO Hyperparameters (from rsl_rl_ppo_cfg.py) ────────────────────────
    num_envs:         int   = 32
    rollout_steps:    int   = 1024
    batch_size:       int   = 512
    n_epochs:         int   = 5           # num_learning_epochs=5
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.2
    clip_value_loss:  bool  = True
    entropy_coef:     float = 0.001
    value_coef:       float = 0.5
    max_grad_norm:    float = 1.0
    learning_rate:    float = 3e-4

    # Adaptive KL-based LR schedule (from rsl_rl_ppo_cfg: schedule="adaptive")
    lr_schedule:      str   = "linear"
    desired_kl:       float = 0.01
    lr_min:           float = 1e-5

    total_timesteps:  int   = 7_000_000

    # ─── Network Architecture (from rsl_rl_ppo_cfg.py) ───────────────────────
    actor_hidden_dims:  Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation:         str             = "elu"
    log_std_init:       float           = 0.0
    log_std_final:      float           = -2.0   # std anneals from exp(0.0)=1.0 to exp(-2.0)=0.135
    ortho_init:         bool            = True

    # ─── Hardware ────────────────────────────────────────────────────────────
    device:           str   = "cuda:0"   # GPU 1 on Hulk PC
    seed:             int   = 42

    # ─── Logging ─────────────────────────────────────────────────────────────
    log_interval:     int   = 1
    save_interval:    int   = 100
    eval_interval:    int   = 25
    eval_episodes:    int   = 10
    run_name:         str   = "ppo_g1_reach_v5"