"""Genesis-based Go2 quadruped env for rsl_rl PPO (5.x VecEnv contract)."""
import math
import torch
import genesis as gs
from tensordict import TensorDict


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda:0"):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = torch.device(device)

        # rsl_rl VecEnv requires env.cfg
        self.cfg = env_cfg

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.dt = 0.02
        self.decimation = 4
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat.unsqueeze(0)).squeeze(0)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=tuple(env_cfg["base_init_pos"]),
                quat=tuple(env_cfg["base_init_quat"]),
            ),
        )
        self.scene.build(n_envs=num_envs)

        self.motor_dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in env_cfg["dof_names"]]
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in env_cfg["dof_names"]],
            device=self.device, dtype=torch.float32,
        )

        # Buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=torch.int32)
        self.commands = torch.zeros((num_envs, self.num_commands), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_lin_vel = torch.zeros((num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=self.device)
        self.projected_gravity = torch.zeros((num_envs, 3), device=self.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(num_envs, 1)
        self.base_pos = torch.zeros((num_envs, 3), device=self.device)
        self.base_quat = torch.zeros((num_envs, 4), device=self.device)
        self.extras = {}

        # Reset everything so the very first get_observations() is valid
        self.reset()

    # ---------- rsl_rl VecEnv contract ----------

    def get_observations(self) -> TensorDict:
        return self._obs_tensordict()

    def get_privileged_observations(self):
        return None

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self._obs_tensordict()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        for _ in range(self.decimation):
            self.scene.step()

        self.episode_length_buf += 1
        self._refresh_robot_state()

        resample_period = int(self.command_cfg["resampling_time_s"] / self.dt)
        time_to_resample = (self.episode_length_buf % resample_period == 0)
        resample_idx = time_to_resample.nonzero(as_tuple=False).flatten()
        if resample_idx.numel() > 0:
            self._resample_commands(resample_idx)

        roll = self._compute_roll(self.base_quat)
        pitch = self._compute_pitch(self.base_quat)
        flip = (roll.abs() > self.env_cfg["termination_if_roll_greater_than"]) | \
               (pitch.abs() > self.env_cfg["termination_if_pitch_greater_than"])
        timeout = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = (flip | timeout).to(torch.int32)

        self._compute_rewards()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self._compute_observations()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return (
            self._obs_tensordict(),
            self.rew_buf,
            self.reset_buf.to(torch.float32),
            self.extras,
        )

    # ---------- internals ----------

    def _obs_tensordict(self) -> TensorDict:
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if envs_idx is None or len(envs_idx) == 0:
            return
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 0

        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        r = self.command_cfg
        self.commands[envs_idx, 0] = _rand(envs_idx, r["lin_vel_x_range"], self.device)
        self.commands[envs_idx, 1] = _rand(envs_idx, r["lin_vel_y_range"], self.device)
        self.commands[envs_idx, 2] = _rand(envs_idx, r["ang_vel_range"], self.device)

    def _refresh_robot_state(self):
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        lin_vel_w = self.robot.get_vel()
        ang_vel_w = self.robot.get_ang()
        self.base_lin_vel[:] = quat_apply_inverse(self.base_quat, lin_vel_w)
        self.base_ang_vel[:] = quat_apply_inverse(self.base_quat, ang_vel_w)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

    def _compute_observations(self):
        self.projected_gravity = quat_apply_inverse(self.base_quat, self.global_gravity)
        cmd_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
        )
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * cmd_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ], dim=-1)

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        cfg = self.reward_cfg
        sigma = cfg["tracking_sigma"]

        lin_err = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=-1)
        self.rew_buf += self.reward_scales["tracking_lin_vel"] * torch.exp(-lin_err / sigma)

        ang_err = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        self.rew_buf += self.reward_scales["tracking_ang_vel"] * torch.exp(-ang_err / sigma)

        self.rew_buf += self.reward_scales["lin_vel_z"] * self.base_lin_vel[:, 2] ** 2
        self.rew_buf += self.reward_scales["ang_vel_xy"] * torch.sum(self.base_ang_vel[:, :2] ** 2, dim=-1)
        self.rew_buf += self.reward_scales["orientation"] * torch.sum(self.projected_gravity[:, :2] ** 2, dim=-1)

        target_h = cfg["base_height_target"]
        self.rew_buf += self.reward_scales["base_height"] * (self.base_pos[:, 2] - target_h) ** 2

        self.rew_buf += self.reward_scales["action_rate"] * torch.sum(
            (self.actions - self.last_actions) ** 2, dim=-1
        )

        sim_idx = [0, 1, 3, 4, 6, 7, 9, 10]
        self.rew_buf += self.reward_scales["similar_to_default"] * torch.sum(
            (self.dof_pos[:, sim_idx] - self.default_dof_pos[sim_idx]) ** 2, dim=-1
        )

        self.rew_buf = self.rew_buf.clamp(min=0.0)

    @staticmethod
    def _compute_roll(q):
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    @staticmethod
    def _compute_pitch(q):
        w, x, y, z = q.unbind(-1)
        sinp = 2 * (w * y - z * x)
        sinp = sinp.clamp(-1.0, 1.0)
        return torch.asin(sinp)


# ---------- helpers ----------

def inv_quat(q):
    """Conjugate (== inverse for unit quat) of (w,x,y,z) batch."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_apply(q, v):
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(inv_quat(q), v)


def _rand(envs_idx, rng, device):
    n = envs_idx.shape[0] if isinstance(envs_idx, torch.Tensor) else len(envs_idx)
    return torch.empty(n, device=device).uniform_(rng[0], rng[1])
