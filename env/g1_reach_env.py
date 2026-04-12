"""
env/g1_reach_env.py — MuJoCo Gymnasium environment for G1 upper-body reaching

Observation pipeline aligned with unitree_rl_lab ObservationsCfg:
  Per-step (42-dim):
    rel_target    (3)  — (target_pos - ee_pos), scaled ×1.0
    joint_pos_rel (13) — joint_pos - default_joint_pos, scaled ×1.0, noise ±0.01
    joint_vel_rel (13) — joint_vel, scaled ×0.05, noise ±1.5
    last_action   (13) — previous action sent to controller

  Stacked over history_length=5 → 210-dim network input
  (matches unitree_rl_lab PolicyCfg history_length=5)

Action pipeline aligned with unitree_rl_lab ActionsCfg:
  JointPositionActionCfg(scale=0.25, use_default_offset=True)
  → actual_target = default_joint_pos + action * 0.25

Reward terms adapted from velocity_env_cfg.py RewardsCfg (locomotion-irrelevant
terms like gait/feet/base_height dropped; reaching task terms added).

Domain randomisation from velocity_env_cfg.py EventCfg:
  - Friction: uniform(0.3, 1.0) on reset
  - Base mass: add uniform(-1, 3) kg on reset
  - Motor strength: multiplicative uniform(0.9, 1.1) on reset
  - Joint init velocity: uniform(-1, 1) rad/s on reset
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, Dict

from config import G1Config


class G1ReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, cfg: G1Config = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or G1Config()
        self.render_mode = render_mode

        # ── Load model ───────────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_xml)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / self.cfg.physics_hz

        # ── Joint / site indices ─────────────────────────────────────────────
        self._ctrl_joint_ids  = self._get_joint_ids()
        self._ctrl_joint_qpos = self._get_joint_qpos_addrs()
        self._ctrl_joint_qvel = self._get_joint_qvel_addrs()
        self._joint_ranges    = self._get_joint_ranges()   # (13, 2)
        self._ee_site_id      = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.ee_site_name
        )
        self._pelvis_body_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self._target_mocap_id = self._get_mocap_id("target_body")

        # Default joint positions (used for joint_pos_rel and action offset)
        self._default_joint_pos = np.zeros(self.cfg.action_dim, dtype=np.float32)

        # ── Domain randomisation state ───────────────────────────────────────
        self._nominal_mass     = None   # set after first model load
        self._motor_strength   = np.ones(self.cfg.action_dim, dtype=np.float32)
        self._nominal_kp       = self._read_kp()

        # ── Observation history (history_length=5, from unitree PolicyCfg) ───
        single_obs = self.cfg.obs_dim   # 42
        self._obs_history = deque(
            [np.zeros(single_obs, dtype=np.float32)] * self.cfg.history_length,
            maxlen=self.cfg.history_length
        )

        # ── Spaces ───────────────────────────────────────────────────────────
        stacked = self.cfg.stacked_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(stacked,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=np.float32
        )

        # ── Episode state ────────────────────────────────────────────────────
        self._step_count   = 0
        self._target_pos   = np.zeros(3, dtype=np.float32)
        self._prev_action  = np.zeros(self.cfg.action_dim, dtype=np.float32)
        self._prev_joint_vel = np.zeros(self.cfg.action_dim, dtype=np.float32)
        self._renderer     = None

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Fix base
        self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        self.data.qvel[:6] = 0.0

        # Domain randomisation (per unitree EventCfg, applied each reset)
        self._apply_domain_randomisation()

        # Randomise initial joint positions near default (±5% of range)
        jlo, jhi = self._joint_ranges[:, 0], self._joint_ranges[:, 1]
        jspan = jhi - jlo
        init_qpos = self._default_joint_pos + self.np_random.uniform(
            -0.05, 0.05, size=self.cfg.action_dim
        ) * jspan
        for i, addr in enumerate(self._ctrl_joint_qpos):
            self.data.qpos[addr] = float(np.clip(init_qpos[i], jlo[i], jhi[i]))

        # Randomise initial joint velocities (per unitree reset_robot_joints)
        lo, hi = self.cfg.dr_joint_vel_range
        for i, addr in enumerate(self._ctrl_joint_qvel):
            self.data.qvel[addr] = float(self.np_random.uniform(lo, hi))

        # Sample target
        self._target_pos = self._sample_target()
        if self._target_mocap_id >= 0:
            self.data.mocap_pos[self._target_mocap_id] = self._target_pos

        mujoco.mj_forward(self.model, self.data)

        # Reset episode state
        self._step_count     = 0
        self._prev_action    = np.zeros(self.cfg.action_dim, dtype=np.float32)
        self._prev_joint_vel = np.zeros(self.cfg.action_dim, dtype=np.float32)
        self._obs_history    = deque(
            [np.zeros(self.cfg.obs_dim, dtype=np.float32)] * self.cfg.history_length,
            maxlen=self.cfg.history_length
        )

        obs  = self._get_stacked_obs()
        info = {"target_pos": self._target_pos.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Action → joint position targets
        # Aligned with unitree ActionsCfg: use_default_offset=True, scale=0.25
        # actual_target = default_joint_pos + action * action_scale
        joint_targets = self._default_joint_pos + action * self.cfg.action_scale
        joint_targets = np.clip(joint_targets, self._joint_ranges[:, 0], self._joint_ranges[:, 1])

        # Apply motor strength domain randomisation to targets
        joint_targets = joint_targets * self._motor_strength + (1 - self._motor_strength) * self._default_joint_pos

        # Step physics
        self.data.ctrl[:] = joint_targets
        for _ in range(self.cfg.control_decimation):
            mujoco.mj_step(self.model, self.data)

        # Fix floating base
        self.data.qpos[:3]  = 0.0
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:6]  = 0.0

        # Build single-step obs and push to history
        single_obs = self._get_single_obs(action)
        self._obs_history.append(single_obs)
        stacked_obs = self._get_stacked_obs()

        # Reward
        reward, info = self._compute_reward(action)

        # Termination
        ee_pos  = self.data.site_xpos[self._ee_site_id].copy()
        dist    = float(np.linalg.norm(ee_pos - self._target_pos))
        success = dist < self.cfg.success_dist
        fall    = self._check_fall()
        terminated = success or fall
        truncated  = self._step_count >= self.cfg.max_episode_steps - 1

        self._step_count    += 1
        self._prev_joint_vel = self._get_joint_vel()
        self._prev_action    = action.copy()

        info.update({
            "dist":       dist,
            "success":    success,
            "fall":       fall,
            "ee_pos":     ee_pos,
            "target_pos": self._target_pos.copy(),
        })
        return stacked_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode in ("human", "rgb_array"):
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            if self.render_mode == "rgb_array":
                return self._renderer.render()
            self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_single_obs(self, last_action: np.ndarray) -> np.ndarray:
        """
        Build 42-dim single-step observation, aligned with unitree_rl_lab PolicyCfg.

        [0:3]   rel_target    = (target_pos - ee_pos) * scale(1.0)
        [3:16]  joint_pos_rel = (joint_pos - default_joint_pos) * scale(1.0) + noise(±0.01)
        [16:29] joint_vel_rel = joint_vel * scale(0.05) + noise(±1.5 * 0.05)
        [29:42] last_action   = previous action (no noise)
        """
        ee_pos = self.data.site_xpos[self._ee_site_id].astype(np.float32).copy()

        # rel_target (3) — scaled ×1.0
        rel_target = (self._target_pos - ee_pos) * self.cfg.obs_scale_rel_target

        # joint_pos_rel (13) — (q - q_default), scaled ×1.0
        jpos = self._get_joint_pos()
        joint_pos_rel = (jpos - self._default_joint_pos) * self.cfg.obs_scale_joint_pos
        joint_pos_rel += self.np_random.uniform(
            -self.cfg.obs_noise_joint_pos, self.cfg.obs_noise_joint_pos,
            size=joint_pos_rel.shape
        ).astype(np.float32)

        # joint_vel_rel (13) — scaled ×0.05, noise added before scaling
        jvel = self._get_joint_vel()
        jvel_noisy = jvel + self.np_random.uniform(
            -self.cfg.obs_noise_joint_vel, self.cfg.obs_noise_joint_vel,
            size=jvel.shape
        ).astype(np.float32)
        joint_vel_rel = jvel_noisy * self.cfg.obs_scale_joint_vel

        obs = np.concatenate([
            rel_target,        # 3
            joint_pos_rel,     # 13
            joint_vel_rel,     # 13
            last_action,       # 13
        ]).astype(np.float32)  # total: 42

        return obs

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack history_length=5 single observations → 210-dim."""
        return np.concatenate(list(self._obs_history), axis=0).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Reward  (adapted from velocity_env_cfg.py RewardsCfg)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        dist   = float(np.linalg.norm(ee_pos - self._target_pos))
        jpos   = self._get_joint_pos()
        jvel   = self._get_joint_vel()

        # Task: dense distance penalty
        r_dist    = self.cfg.rew_weight_dist * dist

        # Task: success bonus
        success   = dist < self.cfg.success_dist
        r_success = self.cfg.rew_weight_success if success else 0.0

        # Alive bonus (mdp.is_alive weight=0.15)
        r_alive   = self.cfg.rew_weight_alive

        # action_rate_l2 (weight=-0.05): penalise change in action
        r_action_rate = self.cfg.rew_weight_action_rate * float(
            np.sum((action - self._prev_action) ** 2)
        )

        # joint_vel_l2 (weight=-0.001)
        r_joint_vel = self.cfg.rew_weight_joint_vel * float(np.sum(jvel ** 2))

        # joint_acc_l2 (weight=-2.5e-7): finite difference approximation
        dt   = (self.cfg.control_decimation / self.cfg.physics_hz)
        jacc = (jvel - self._prev_joint_vel) / (dt + 1e-8)
        r_joint_acc = self.cfg.rew_weight_joint_acc * float(np.sum(jacc ** 2))

        # energy (weight=-2e-5): |vel| * |torque|, approx via ctrl * kp
        torque_approx = np.abs(self.data.ctrl[:self.cfg.action_dim]) * np.abs(jvel)
        r_energy = self.cfg.rew_weight_energy * float(np.sum(torque_approx))

        # dof_pos_limits (weight=-5.0): penalise joints near limits
        jlo, jhi   = self._joint_ranges[:, 0], self._joint_ranges[:, 1]
        margin      = 0.05 * (jhi - jlo)
        over_lo     = np.maximum(0.0, jlo + margin - jpos)
        over_hi     = np.maximum(0.0, jpos - (jhi - margin))
        r_dof_lim   = self.cfg.rew_weight_dof_limits * float(np.sum(over_lo + over_hi))

        # joint_deviation_arms (weight=-0.1): L1 from default joint pos
        r_joint_dev = self.cfg.rew_weight_joint_dev * float(
            np.sum(np.abs(jpos - self._default_joint_pos))
        )

        # Fall penalty
        fall    = self._check_fall()
        r_fall  = self.cfg.rew_weight_fall if fall else 0.0

        reward = (
            r_dist + r_success + r_alive
            + r_action_rate + r_joint_vel + r_joint_acc
            + r_energy + r_dof_lim + r_joint_dev
            + r_fall
        )

        info = {
            "r_dist":        r_dist,
            "r_success":     r_success,
            "r_alive":       r_alive,
            "r_action_rate": r_action_rate,
            "r_joint_vel":   r_joint_vel,
            "r_joint_acc":   r_joint_acc,
            "r_energy":      r_energy,
            "r_dof_lim":     r_dof_lim,
            "r_joint_dev":   r_joint_dev,
            "r_fall":        r_fall,
        }
        return float(reward), info

    # ─────────────────────────────────────────────────────────────────────────
    # Domain Randomisation  (from velocity_env_cfg.py EventCfg)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_domain_randomisation(self):
        """
        Applied each reset, mirroring unitree_rl_lab EventCfg:
          - physics_material  → friction randomisation
          - add_base_mass     → torso mass perturbation
          - motor strength    → multiplicative (not in unitree but standard practice)
          - reset_robot_joints velocity handled in reset()
        """
        # Friction (physics_material event: static/dynamic uniform(0.3, 1.0))
        lo, hi = self.cfg.dr_friction_range
        new_friction = float(self.np_random.uniform(lo, hi))
        for i in range(self.model.ngeom):
            self.model.geom_friction[i, 0] = new_friction  # sliding friction

        # Base mass (add_base_mass event: add uniform(-1, 3) kg to torso)
        if self._nominal_mass is None:
            self._nominal_mass = float(self.model.body_mass[self._pelvis_body_id])
        lo, hi = self.cfg.dr_mass_range
        mass_offset = float(self.np_random.uniform(lo, hi))
        self.model.body_mass[self._pelvis_body_id] = self._nominal_mass + mass_offset

        # Motor strength (multiplicative, standard sim-to-real practice)
        lo, hi = self.cfg.dr_motor_strength_range
        self._motor_strength = self.np_random.uniform(lo, hi, size=self.cfg.action_dim).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_joint_pos(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[addr] for addr in self._ctrl_joint_qpos], dtype=np.float32
        )

    def _get_joint_vel(self) -> np.ndarray:
        return np.array(
            [self.data.qvel[addr] for addr in self._ctrl_joint_qvel], dtype=np.float32
        )

    def _check_fall(self) -> bool:
        return float(self.data.xpos[self._pelvis_body_id][2]) < self.cfg.fall_height_threshold

    def _sample_target(self) -> np.ndarray:
        cx, cy, cz = self.cfg.target_center
        r = self.cfg.target_radius
        while True:
            p = self.np_random.uniform(-1, 1, size=3) * r
            if np.linalg.norm(p) <= r:
                t = np.array([cx + p[0], cy + p[1], cz + p[2]], dtype=np.float32)
                t[2] = max(t[2], 0.3)
                return t

    def _read_kp(self) -> np.ndarray:
        """Read PD position gains for controlled actuators."""
        kp = np.zeros(self.cfg.action_dim, dtype=np.float32)
        for i in range(min(self.cfg.action_dim, self.model.nu)):
            kp[i] = self.model.actuator_gainprm[i, 0]
        return kp

    def _get_joint_ids(self):
        ids = []
        for name in self.cfg.controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MJCF.")
            ids.append(jid)
        return ids

    def _get_joint_qpos_addrs(self):
        return [self.model.jnt_qposadr[jid] for jid in self._ctrl_joint_ids]

    def _get_joint_qvel_addrs(self):
        return [self.model.jnt_dofadr[jid] for jid in self._ctrl_joint_ids]

    def _get_joint_ranges(self) -> np.ndarray:
        return np.array(
            [self.model.jnt_range[jid] for jid in self._ctrl_joint_ids], dtype=np.float64
        )

    def _get_mocap_id(self, body_name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return -1
        return int(self.model.body_mocapid[bid])
