# Go2 PPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a Unitree Go2 quadruped to follow velocity commands using PPO (rsl_rl) on top of the Genesis physics engine, then play the policy back in a viewer.

**Architecture:** A `Go2Env` class wraps Genesis to expose an rsl_rl-compatible `VecEnv` with 4096 parallel envs on GPU. A separate `configs/go2_config.py` returns the env/obs/reward/command dicts plus the PPO `train_cfg`. `scripts/train.py` runs PPO via `rsl_rl.runners.OnPolicyRunner`; `scripts/eval.py` loads a checkpoint and renders the policy.

**Tech Stack:** Python 3.10+, PyTorch 2.x (CUDA), `genesis-world`, `rsl-rl-lib` 2.x, tensorboard, pytest for contract tests.

**Spec reference:** [`docs/superpowers/specs/2026-05-12-go2-ppo-training-design.md`](../specs/2026-05-12-go2-ppo-training-design.md)

---

## File Map

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned deps (`genesis-world`, `rsl-rl-lib`, `tensorboard`, `pytest`) |
| `.gitignore` | Ignore `logs/`, caches, venv |
| `envs/__init__.py` | Re-export `Go2Env` |
| `envs/go2_env.py` | `Go2Env` class: scene/robot setup, reset, step, observations, rewards, commands, termination |
| `configs/__init__.py` | Re-export `get_cfgs`, `get_train_cfg` |
| `configs/go2_config.py` | `get_cfgs()` and `get_train_cfg()` factory functions |
| `scripts/train.py` | CLI training entry — instantiates env, OnPolicyRunner, calls `learn` |
| `scripts/eval.py` | CLI eval entry — loads checkpoint, runs inference policy in viewer |
| `tests/test_configs.py` | Sanity-check config shapes/keys |
| `tests/test_go2_env.py` | Contract tests: env builds, reset/step return correct shapes & dtypes |
| `tests/test_train_smoke.py` | Smoke test: 2 PPO iterations × 64 envs runs to completion |

---

## Task 1: Project scaffolding & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `envs/__init__.py`, `configs/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`

- [ ] **Step 1.1: Create `requirements.txt`**

```
genesis-world>=0.2.1
rsl-rl-lib>=2.0.0
tensorboard>=2.15
numpy
pytest>=7.0
```

- [ ] **Step 1.2: Create `.gitignore`**

```
logs/
__pycache__/
*.pyc
*.pkl
.venv/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 1.3: Create package `__init__.py` files**

Create empty files at:
- `envs/__init__.py`
- `configs/__init__.py`
- `scripts/__init__.py`
- `tests/__init__.py`

- [ ] **Step 1.4: Install dependencies**

Run:
```bash
pip install -r requirements.txt
```

Expected: install completes without error. If pip cannot find `genesis-world`, instruct the user to verify their Python version is 3.10+ and Python index is reachable.

- [ ] **Step 1.5: Verify imports**

Run:
```bash
python -c "import genesis as gs; import rsl_rl; import torch; print('genesis:', gs.__version__ if hasattr(gs, '__version__') else 'ok'); print('cuda:', torch.cuda.is_available())"
```

Expected output: prints versions, `cuda: True`.

If `cuda: False`, halt and check that the existing torch install is the `+cu*` build.

- [ ] **Step 1.6: Commit**

```bash
git add requirements.txt .gitignore envs/__init__.py configs/__init__.py scripts/__init__.py tests/__init__.py
git commit -m "chore: scaffold project layout and dependencies"
```

---

## Task 2: Genesis hardware smoke test

Verify Genesis can load and step Go2 on this GPU before we build the env around it.

**Files:**
- Create: `tests/test_genesis_smoke.py`

- [ ] **Step 2.1: Write the smoke test**

```python
# tests/test_genesis_smoke.py
"""Sanity-check: Genesis can load Go2 URDF and step physics on GPU."""
import pytest
import torch


def test_genesis_loads_and_steps_go2():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02),
        viewer_options=gs.options.ViewerOptions(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.42),
        ),
    )
    scene.build(n_envs=2)

    for _ in range(10):
        scene.step()

    pos = robot.get_pos()
    assert pos.shape == (2, 3)
    assert torch.isfinite(pos).all()
```

- [ ] **Step 2.2: Run the test**

Run: `pytest tests/test_genesis_smoke.py -v`

Expected: **PASS**. If it fails because the bundled Go2 URDF path differs in this `genesis-world` version, inspect with:
```bash
python -c "import genesis, os; print(os.path.dirname(genesis.__file__))"
```
then locate `go2.urdf` under that tree and update the `file=` argument. Do NOT proceed until this test passes — every downstream task depends on the URDF path being correct.

- [ ] **Step 2.3: Commit**

```bash
git add tests/test_genesis_smoke.py
git commit -m "test: smoke-test Genesis Go2 URDF load and step"
```

---

## Task 3: Config — `get_cfgs()`

**Files:**
- Create: `configs/go2_config.py`
- Create: `tests/test_configs.py`

- [ ] **Step 3.1: Write the failing test**

```python
# tests/test_configs.py
from configs.go2_config import get_cfgs


def test_get_cfgs_returns_four_dicts_with_expected_keys():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    assert env_cfg["num_actions"] == 12
    assert env_cfg["episode_length_s"] == 20.0
    assert env_cfg["action_scale"] == 0.25
    assert "default_joint_angles" in env_cfg
    assert len(env_cfg["default_joint_angles"]) == 12
    assert env_cfg["dof_names"] and len(env_cfg["dof_names"]) == 12

    assert obs_cfg["num_obs"] == 45
    assert "obs_scales" in obs_cfg

    assert reward_cfg["tracking_sigma"] == 0.25
    assert reward_cfg["base_height_target"] == 0.42
    expected_reward_terms = {
        "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z",
        "ang_vel_xy", "orientation", "base_height",
        "action_rate", "similar_to_default",
    }
    assert set(reward_cfg["reward_scales"].keys()) == expected_reward_terms

    assert command_cfg["num_commands"] == 3
    assert command_cfg["lin_vel_x_range"] == [-1.0, 1.0]
    assert command_cfg["lin_vel_y_range"] == [-1.0, 1.0]
    assert command_cfg["ang_vel_range"] == [-1.0, 1.0]
    assert command_cfg["resampling_time_s"] == 10.0
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `pytest tests/test_configs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'configs.go2_config'`.

- [ ] **Step 3.3: Implement `get_cfgs()`**

Create `configs/go2_config.py`:

```python
"""Config factory for Go2 velocity-tracking PPO training."""


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "dof_names": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
        },
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 1.5708,    # pi/2
        "termination_if_pitch_greater_than": 1.5708,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],         # (w, x, y, z)
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.42,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-1.0, 1.0],
        "resampling_time_s": 10.0,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `pytest tests/test_configs.py::test_get_cfgs_returns_four_dicts_with_expected_keys -v`
Expected: PASS.

- [ ] **Step 3.5: Commit**

```bash
git add configs/go2_config.py tests/test_configs.py
git commit -m "feat(configs): add get_cfgs() for Go2 velocity-tracking env"
```

---

## Task 4: Config — `get_train_cfg()` (PPO hyperparams)

**Files:**
- Modify: `configs/go2_config.py`
- Modify: `tests/test_configs.py`

- [ ] **Step 4.1: Add the failing test**

Append to `tests/test_configs.py`:

```python
from configs.go2_config import get_train_cfg


def test_get_train_cfg_has_ppo_and_runner_sections():
    cfg = get_train_cfg("foo", max_iterations=500)

    assert cfg["runner"]["max_iterations"] == 500
    assert cfg["runner"]["experiment_name"] == "foo"
    assert cfg["runner"]["save_interval"] == 50

    assert cfg["algorithm"]["gamma"] == 0.99
    assert cfg["algorithm"]["lam"] == 0.95
    assert cfg["algorithm"]["clip_param"] == 0.2
    assert cfg["algorithm"]["num_learning_epochs"] == 5
    assert cfg["algorithm"]["num_mini_batches"] == 4
    assert cfg["algorithm"]["learning_rate"] == 1e-3
    assert cfg["algorithm"]["schedule"] == "adaptive"
    assert cfg["algorithm"]["desired_kl"] == 0.01

    assert cfg["policy"]["actor_hidden_dims"] == [512, 256, 128]
    assert cfg["policy"]["critic_hidden_dims"] == [512, 256, 128]
    assert cfg["policy"]["activation"] == "elu"
    assert cfg["policy"]["init_noise_std"] == 1.0

    assert cfg["seed"] == 1
    assert cfg["num_steps_per_env"] == 24
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_configs.py::test_get_train_cfg_has_ppo_and_runner_sections -v`
Expected: FAIL with `ImportError: cannot import name 'get_train_cfg'`.

- [ ] **Step 4.3: Implement `get_train_cfg()`**

Append to `configs/go2_config.py`:

```python
def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `pytest tests/test_configs.py -v`
Expected: both tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add configs/go2_config.py tests/test_configs.py
git commit -m "feat(configs): add get_train_cfg() with PPO hyperparameters"
```

---

## Task 5: `Go2Env` — scaffolding (init only)

**Files:**
- Create: `envs/go2_env.py`
- Modify: `envs/__init__.py`
- Create: `tests/test_go2_env.py`

- [ ] **Step 5.1: Write the failing test**

```python
# tests/test_go2_env.py
import pytest
import torch


@pytest.fixture(scope="module")
def env():
    import genesis as gs
    from configs.go2_config import get_cfgs
    from envs.go2_env import Go2Env

    gs.init(backend=gs.gpu, logging_level="warning")
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = Go2Env(
        num_envs=4,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    return env


def test_env_exposes_required_attributes(env):
    assert env.num_envs == 4
    assert env.num_obs == 45
    assert env.num_actions == 12
    assert env.max_episode_length > 0
    assert env.device.type == "cuda"
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `pytest tests/test_go2_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs.go2_env'`.

- [ ] **Step 5.3: Implement env scaffolding**

Create `envs/go2_env.py`:

```python
"""Genesis-based Go2 quadruped env for rsl_rl PPO."""
import math
import torch
import genesis as gs


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda:0"):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = torch.device(device)

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

        # Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
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

        # DOF indices in URDF order
        self.motor_dofs = [self.robot.get_joint(n).dof_idx_local for n in env_cfg["dof_names"]]
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # Default joint pose tensor (in env_cfg["dof_names"] order)
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
        self.extras = {"observations": {}}

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None


def inv_quat(q):
    """Conjugate (inverse for unit quat) of (w,x,y,z) batch."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
```

Update `envs/__init__.py`:

```python
from envs.go2_env import Go2Env

__all__ = ["Go2Env"]
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `pytest tests/test_go2_env.py::test_env_exposes_required_attributes -v`
Expected: PASS.

If `self.robot.get_joint(...).dof_idx_local` does not exist on this Genesis version, check the actual Go2 example under the installed `genesis` package (`python -c "import genesis, os; print(os.path.dirname(genesis.__file__))"` then look for `examples/locomotion/go2_env.py`) and mirror the joint-index lookup it uses.

- [ ] **Step 5.5: Commit**

```bash
git add envs/__init__.py envs/go2_env.py tests/test_go2_env.py
git commit -m "feat(env): scaffold Go2Env with scene, robot, and buffers"
```

---

## Task 6: `Go2Env.reset()` and quaternion helpers

**Files:**
- Modify: `envs/go2_env.py`
- Modify: `tests/test_go2_env.py`

- [ ] **Step 6.1: Add the failing test**

Append to `tests/test_go2_env.py`:

```python
def test_reset_returns_obs_of_expected_shape_and_clears_dones(env):
    obs, extras = env.reset()
    assert obs.shape == (env.num_envs, env.num_obs)
    assert obs.dtype == torch.float32
    assert torch.isfinite(obs).all()
    assert (env.reset_buf == 0).all()
    assert (env.episode_length_buf == 0).all()
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `pytest tests/test_go2_env.py::test_reset_returns_obs_of_expected_shape_and_clears_dones -v`
Expected: FAIL with `AttributeError: 'Go2Env' object has no attribute 'reset'`.

- [ ] **Step 6.3: Implement quaternion helpers and `reset`**

Append to `envs/go2_env.py` (after the `Go2Env` class, replacing `inv_quat` already there with the fuller helpers):

```python
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_apply_inverse(q, v):
    """Rotate v by the inverse of unit quaternion q (both in body frame)."""
    q_inv = inv_quat(q)
    return quat_apply(q_inv, v)


def quat_apply(q, v):
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)
```

Then add these methods to `Go2Env`:

```python
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # Reset DoFs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # Reset base pose
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 0

        # Resample commands
        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        r = self.command_cfg
        self.commands[envs_idx, 0] = _rand(envs_idx, r["lin_vel_x_range"], self.device)
        self.commands[envs_idx, 1] = _rand(envs_idx, r["lin_vel_y_range"], self.device)
        self.commands[envs_idx, 2] = _rand(envs_idx, r["ang_vel_range"], self.device)

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # Populate obs once so the runner sees a valid first obs
        self._compute_observations()
        return self.obs_buf, self.extras

    def _compute_observations(self):
        # Project gravity into body frame
        self.projected_gravity = quat_apply_inverse(self.base_quat, self.global_gravity)
        # commands scaled
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
```

And add this module-level helper near the other helpers:

```python
def _rand(envs_idx, rng, device):
    n = envs_idx.shape[0] if isinstance(envs_idx, torch.Tensor) else len(envs_idx)
    return torch.empty(n, device=device).uniform_(rng[0], rng[1])
```

- [ ] **Step 6.4: Run test to verify it passes**

Run: `pytest tests/test_go2_env.py -v`
Expected: 2 PASS.

- [ ] **Step 6.5: Commit**

```bash
git add envs/go2_env.py tests/test_go2_env.py
git commit -m "feat(env): implement reset, command resampling, observation builder"
```

---

## Task 7: `Go2Env.step()` — actions, simulation, terminations

**Files:**
- Modify: `envs/go2_env.py`
- Modify: `tests/test_go2_env.py`

- [ ] **Step 7.1: Add the failing test**

Append to `tests/test_go2_env.py`:

```python
def test_step_returns_correct_tuple_and_shapes(env):
    env.reset()
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, rewards, dones, info = env.step(actions)

    assert obs.shape == (env.num_envs, env.num_obs)
    assert rewards.shape == (env.num_envs,)
    assert dones.shape == (env.num_envs,)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(rewards).all()
    assert isinstance(info, dict)
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `pytest tests/test_go2_env.py::test_step_returns_correct_tuple_and_shapes -v`
Expected: FAIL with `AttributeError: ...has no attribute 'step'`.

- [ ] **Step 7.3: Implement `step` (no rewards yet — return zeros)**

Add these methods to `Go2Env`:

```python
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        for _ in range(self.decimation):
            self.scene.step()

        self.episode_length_buf += 1
        self._refresh_robot_state()

        # Resample commands periodically
        time_to_resample = (
            self.episode_length_buf
            % int(self.command_cfg["resampling_time_s"] / self.dt) == 0
        )
        resample_idx = time_to_resample.nonzero(as_tuple=False).flatten()
        if resample_idx.numel() > 0:
            self._resample_commands(resample_idx)

        # Terminations
        roll = self._compute_roll(self.base_quat)
        pitch = self._compute_pitch(self.base_quat)
        flip = (roll.abs() > self.env_cfg["termination_if_roll_greater_than"]) | \
               (pitch.abs() > self.env_cfg["termination_if_pitch_greater_than"])
        timeout = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = (flip | timeout).to(torch.int32)

        # Compute rewards (zero for now — implemented in Task 8)
        self.rew_buf[:] = 0.0

        # Reset envs that terminated
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # Update obs and bookkeeping
        self._compute_observations()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf.to(torch.float32), self.extras

    def _refresh_robot_state(self):
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        # Body-frame velocities
        lin_vel_w = self.robot.get_vel()
        ang_vel_w = self.robot.get_ang()
        self.base_lin_vel[:] = quat_apply_inverse(self.base_quat, lin_vel_w)
        self.base_ang_vel[:] = quat_apply_inverse(self.base_quat, ang_vel_w)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

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
```

- [ ] **Step 7.4: Run test to verify it passes**

Run: `pytest tests/test_go2_env.py -v`
Expected: 3 PASS.

If the Genesis API used here (`get_vel`, `get_ang`, `get_dofs_position`, `control_dofs_position`) differs in your installed version, consult `examples/locomotion/go2_env.py` in the installed `genesis` package — the public API names there are authoritative for the version you have.

- [ ] **Step 7.5: Commit**

```bash
git add envs/go2_env.py tests/test_go2_env.py
git commit -m "feat(env): implement step with simulation, termination, and resampling"
```

---

## Task 8: Reward computation

**Files:**
- Modify: `envs/go2_env.py`
- Modify: `tests/test_go2_env.py`

- [ ] **Step 8.1: Add the failing test**

Append to `tests/test_go2_env.py`:

```python
def test_rewards_are_finite_and_nonzero_when_off_target(env):
    env.reset()
    # Force a clear lin_vel_x command of 1.0; robot is stationary, so tracking
    # reward should be < 1.0 but > 0 (exp kernel never zero with sigma=0.25).
    env.commands[:, 0] = 1.0
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    _, rewards, _, _ = env.step(actions)
    assert torch.isfinite(rewards).all()
    assert (rewards >= 0).all()       # rsl_rl convention: clipped to >= 0
```

- [ ] **Step 8.2: Run test to verify it fails**

Run: `pytest tests/test_go2_env.py::test_rewards_are_finite_and_nonzero_when_off_target -v`
Expected: FAIL because `rew_buf` is hard-coded to 0 (test would pass on `>= 0` but fail on being meaningfully populated — adjust to also assert `rewards.sum() > 0` after switching the test to expect non-zero shaping):

Update the assertion to require at least some positive shaping:

```python
    assert rewards.sum() > 0    # tracking reward never zero
```

Re-run; it should now FAIL.

- [ ] **Step 8.3: Implement reward terms**

Replace `self.rew_buf[:] = 0.0` in `step()` with a call to `self._compute_rewards()`, and add this method to `Go2Env`:

```python
    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        cfg = self.reward_cfg
        sigma = cfg["tracking_sigma"]

        # 1. tracking_lin_vel
        lin_err = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=-1)
        self.rew_buf += self.reward_scales["tracking_lin_vel"] * torch.exp(-lin_err / sigma)

        # 2. tracking_ang_vel
        ang_err = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        self.rew_buf += self.reward_scales["tracking_ang_vel"] * torch.exp(-ang_err / sigma)

        # 3. lin_vel_z (penalty)
        self.rew_buf += self.reward_scales["lin_vel_z"] * self.base_lin_vel[:, 2] ** 2

        # 4. ang_vel_xy (penalty)
        self.rew_buf += self.reward_scales["ang_vel_xy"] * torch.sum(self.base_ang_vel[:, :2] ** 2, dim=-1)

        # 5. orientation (penalty)
        self.rew_buf += self.reward_scales["orientation"] * torch.sum(self.projected_gravity[:, :2] ** 2, dim=-1)

        # 6. base_height (penalty)
        target_h = cfg["base_height_target"]
        self.rew_buf += self.reward_scales["base_height"] * (self.base_pos[:, 2] - target_h) ** 2

        # 7. action_rate (penalty)
        self.rew_buf += self.reward_scales["action_rate"] * torch.sum(
            (self.actions - self.last_actions) ** 2, dim=-1
        )

        # 8. similar_to_default — hip+thigh joints only (indices 0,1,3,4,6,7,9,10)
        sim_idx = [0, 1, 3, 4, 6, 7, 9, 10]
        self.rew_buf += self.reward_scales["similar_to_default"] * torch.sum(
            (self.dof_pos[:, sim_idx] - self.default_dof_pos[sim_idx]) ** 2, dim=-1
        )

        # Clip to non-negative (rsl_rl convention)
        self.rew_buf = self.rew_buf.clamp(min=0.0)
```

- [ ] **Step 8.4: Run test to verify it passes**

Run: `pytest tests/test_go2_env.py -v`
Expected: 4 PASS.

- [ ] **Step 8.5: Commit**

```bash
git add envs/go2_env.py tests/test_go2_env.py
git commit -m "feat(env): implement 8-term reward shaping for velocity tracking"
```

---

## Task 9: Training script

**Files:**
- Create: `scripts/train.py`
- Create: `tests/test_train_smoke.py`

- [ ] **Step 9.1: Write the smoke test**

```python
# tests/test_train_smoke.py
"""End-to-end smoke: 2 PPO iterations × 64 envs runs to completion."""
import os
import subprocess
import sys
import pytest


def test_train_smoke(tmp_path):
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cmd = [
        sys.executable, "scripts/train.py",
        "--exp_name", "smoke",
        "--num_envs", "64",
        "--max_iterations", "2",
        "--headless",
        "--log_root", str(log_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (log_dir / "smoke").exists()
    # At least one .pt model file was written
    pt_files = list((log_dir / "smoke").rglob("*.pt"))
    assert pt_files, "no .pt checkpoint produced"
```

- [ ] **Step 9.2: Run test to verify it fails**

Run: `pytest tests/test_train_smoke.py -v`
Expected: FAIL because `scripts/train.py` doesn't exist.

- [ ] **Step 9.3: Implement `scripts/train.py`**

```python
"""Train Go2 velocity-tracking policy with PPO (rsl_rl) on Genesis."""
import argparse
import os
import pickle
import sys

# Ensure repo root is on PYTHONPATH when invoked as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from configs.go2_config import get_cfgs, get_train_cfg
from envs.go2_env import Go2Env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="go2_walk")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--log_root", type=str, default="logs")
    return p.parse_args()


def main():
    args = parse_args()
    gs.init(backend=gs.gpu, logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    train_cfg["seed"] = args.seed

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
    )

    log_dir = os.path.join(args.log_root, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.4: Run smoke test**

Run: `pytest tests/test_train_smoke.py -v -s`
Expected: PASS within ~10 minutes (cold Genesis init + 2 iters at 64 envs).

If `OnPolicyRunner.__init__` complains about missing keys in `train_cfg`, copy the missing key from the installed `rsl_rl` source: `python -c "import rsl_rl, os; print(os.path.dirname(rsl_rl.__file__))"`. Then add it to `get_train_cfg()` and re-run.

- [ ] **Step 9.5: Commit**

```bash
git add scripts/train.py tests/test_train_smoke.py
git commit -m "feat(scripts): add PPO training entry point with smoke test"
```

---

## Task 10: Evaluation / playback script

**Files:**
- Create: `scripts/eval.py`
- Modify: `tests/test_train_smoke.py` (add eval-load assertion)

- [ ] **Step 10.1: Add an eval-load smoke test**

Append to `tests/test_train_smoke.py`:

```python
def test_eval_can_load_smoke_checkpoint(tmp_path):
    """After training, eval.py should load the checkpoint and step the policy."""
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # First train a tiny model
    subprocess.run([
        sys.executable, "scripts/train.py",
        "--exp_name", "evalsmoke",
        "--num_envs", "64",
        "--max_iterations", "2",
        "--headless",
        "--log_root", str(log_dir),
    ], cwd=repo_root, env=env, check=True, capture_output=True, timeout=600)

    # Then eval it for 5 steps headless
    result = subprocess.run([
        sys.executable, "scripts/eval.py",
        "--exp_name", "evalsmoke",
        "--ckpt", "last",
        "--max_steps", "5",
        "--headless",
        "--log_root", str(log_dir),
    ], cwd=repo_root, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `pytest tests/test_train_smoke.py::test_eval_can_load_smoke_checkpoint -v`
Expected: FAIL because `scripts/eval.py` doesn't exist.

- [ ] **Step 10.3: Implement `scripts/eval.py`**

```python
"""Play back a trained Go2 policy in the Genesis viewer."""
import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from envs.go2_env import Go2Env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="last",
                   help='Iteration number, or "last" for the most recent.')
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0,
                   help="0 = run forever; otherwise stop after N steps (for tests).")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--log_root", type=str, default="logs")
    return p.parse_args()


def resolve_checkpoint(log_dir, ckpt):
    if ckpt == "last":
        candidates = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
        if not candidates:
            raise FileNotFoundError(f"no model_*.pt found under {log_dir}")
        return candidates[-1]
    path = os.path.join(log_dir, f"model_{ckpt}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def main():
    args = parse_args()
    gs.init(backend=gs.gpu, logging_level="warning")

    log_dir = os.path.join(args.log_root, args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.load(resolve_checkpoint(log_dir, args.ckpt))
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    step = 0
    with torch.no_grad():
        while args.max_steps == 0 or step < args.max_steps:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.4: Run eval smoke test**

Run: `pytest tests/test_train_smoke.py::test_eval_can_load_smoke_checkpoint -v -s`
Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add scripts/eval.py tests/test_train_smoke.py
git commit -m "feat(scripts): add eval.py for trained policy playback"
```

---

## Task 11: Full training run

This task is operational, not code. It validates the success criteria from the design spec.

- [ ] **Step 11.1: Launch full training**

Run:
```bash
python scripts/train.py --exp_name go2_walk --max_iterations 500 --headless
```

Expected: terminal prints PPO iteration logs; completes in 5–15 minutes. `logs/go2_walk/model_500.pt` exists at the end.

- [ ] **Step 11.2: Inspect Tensorboard curves**

Run (in a separate terminal):
```bash
tensorboard --logdir logs/ --port 6006
```

Open `http://localhost:6006` and confirm:
- `Train/mean_reward` is increasing and finishes above **8.0**.
- `Train/mean_episode_length` is above **800** by the end.
- `Loss/surrogate` is finite and trending down.

If `mean_reward` is stuck below 5.0 after 500 iterations, do NOT proceed. Open the systematic-debugging skill and investigate (most likely cause: a sign error in a reward term or wrong gravity projection; re-verify `_compute_observations` and `_compute_rewards`).

- [ ] **Step 11.3: Visual eval**

Run:
```bash
python scripts/eval.py --exp_name go2_walk --ckpt last
```

Expected: Genesis viewer opens, Go2 walks forward smoothly without falling for at least 20 seconds.

- [ ] **Step 11.4: Commit the design-validation note**

If you want to capture the training results in git, add a brief note (optional) and commit. Otherwise this task ends here — there is nothing new under version control.

---

## Verification Summary

When the plan is done you should have:

- `pytest tests/ -v` → all tests pass (configs + env contract + train smoke + eval smoke).
- `logs/go2_walk/model_500.pt` produced after a 500-iter run.
- Tensorboard curves matching the success thresholds (`mean_reward > 8.0`, `mean_episode_length > 800`).
- Eval viewer shows a walking Go2.

---

## Self-Review

- **Spec coverage:** All sections of the spec are mapped to tasks: Section 4 architecture → Tasks 5–7; Section 5.1 Go2Env → Tasks 5–8; Section 5.2 configs → Tasks 3–4; Section 5.3 train.py → Task 9; Section 5.4 eval.py → Task 10; Section 6 rewards → Task 8; Section 7 PPO config → Task 4; Section 9 verification → Tasks 9, 10, 11.
- **Placeholder scan:** No TBD/TODO/"similar to". Every code step has full code.
- **Type consistency:** `Go2Env` signature `(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, device)` is identical in Tasks 5, 9, 10. Config keys (`reward_scales` with 8 entries, `obs_cfg["num_obs"] == 45`, `command_cfg["num_commands"] == 3`) match between tests and implementation. `step` returns `(obs, rewards, dones, extras)` in Tasks 7 and onwards.
- **Known fragility:** Tasks 5 and 7 explicitly call out that Genesis public API names (`get_vel`, `control_dofs_position`, `dof_idx_local`, URDF path) may differ slightly between `genesis-world` versions; both tasks point at `examples/locomotion/go2_env.py` in the installed package as the authoritative reference. This is intentional — pinning that exactly would require running install first.
