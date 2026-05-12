# Go2 Quadruped PPO Training in Genesis — Design Spec

**Date:** 2026-05-12
**Author:** brainstorming session
**Status:** Approved

## 1. Goal

Train a Unitree Go2 quadruped robot to perform **velocity tracking** locomotion (follow `[v_x, v_y, ω_z]` commands) using **PPO** in the **Genesis** GPU-parallel physics engine. Deliver a full end-to-end pipeline: training, monitoring, and policy playback in a viewer.

**Success criteria:**
- `mean_reward > 8.0` and `mean_episode_length > 800` (out of 1000) after ~500 PPO iterations.
- Eval playback shows Go2 walking smoothly without falling for a sustained command (e.g., `v_x = 0.5 m/s`).
- Total wall-clock training time: under 15 minutes on an RTX 5060 Ti.

## 2. Non-Goals (out of scope for v1)

- Curriculum learning over command ranges.
- Sim-to-real domain randomization (terrain heightmaps, mass/friction perturbation, control latency).
- Privileged / asymmetric critic observations.
- Multi-robot support (A1, ANYmal, custom URDFs).
- ONNX export for hardware deployment.
- Rough terrain, stairs, or obstacle traversal.

YAGNI — these can be added in future iterations.

## 3. Environment

### Hardware target
- GPU: NVIDIA RTX 5060 Ti, 16 GB VRAM
- CUDA 13.0, PyTorch 2.11.0+cu130
- Genesis backend: `gs.gpu`

### Software stack
- `genesis-world>=0.2.1`
- `rsl-rl-lib>=2.0.0`
- `tensorboard`
- `numpy`
- `torch` (already installed: 2.11.0+cu130)

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     train.py (entry)                        │
│  - parse args, set seed, init logger                        │
│  - create Go2Env (4096 parallel envs on GPU)                │
│  - create OnPolicyRunner (rsl_rl PPO)                       │
│  - runner.learn(num_iterations)                             │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌──────────────────┐               ┌────────────────────┐
│  Go2Env          │◀─obs/reward──▶│  rsl_rl PPO        │
│  (Genesis)       │               │  ActorCritic MLP   │
│  - 4096 envs     │──actions────▶ │  [512,256,128]     │
│  - URDF Go2      │               │  GAE λ=0.95        │
│  - reset/step    │               │  γ=0.99            │
└──────────────────┘               └────────────────────┘
                                           │
                                           ▼
                                  ┌────────────────────┐
                                  │  logs/<exp>/       │
                                  │  - model_*.pt      │
                                  │  - tensorboard     │
                                  └────────────────────┘
```

### Data flow per iteration

1. `env.reset()` returns `obs: [num_envs, 45]` on GPU.
2. Policy forward → `actions: [num_envs, 12]` (joint position offsets).
3. Genesis simulates one control step: physics @ 200 Hz, `decimation=4`, control @ 50 Hz, `dt=0.02s`.
4. Compute rewards, dones, infos → tuple returned to runner.
5. Buffer rollout of 24 steps → PPO update: 5 epochs × 4 mini-batches.
6. Repeat for `max_iterations`.

## 5. Components

### 5.1 `envs/go2_env.py` — Go2Env

**Robot:** Unitree Go2, 12 DoF (4 legs × {hip, thigh, calf}).
**Control:** PD position control via Genesis built-in (`Kp=20`, `Kd=0.5`).

**Action space** — `[num_envs, 12]`, scaled position offsets:

```
target_dof_pos = default_dof_pos + action * action_scale   # action_scale = 0.25
```

**Observation space — 45 dims:**

| # | Component                          | Dims | Scale       |
|---|------------------------------------|------|-------------|
| 1 | base angular velocity (body frame) | 3    | 0.25        |
| 2 | projected gravity (body frame)     | 3    | 1.0         |
| 3 | velocity commands [vx, vy, wz]     | 3    | (1, 1, 0.25)|
| 4 | joint positions − default          | 12   | 1.0         |
| 5 | joint velocities                   | 12   | 0.05        |
| 6 | last actions                       | 12   | 1.0         |

Base linear velocity is **not** in the observation (standard rsl_rl convention; policy must infer it from joint dynamics).

**Commands** — resampled per env every 10 s:
- `lin_vel_x ∈ [-1.0, 1.0]` m/s
- `lin_vel_y ∈ [-1.0, 1.0]` m/s
- `ang_vel_z ∈ [-1.0, 1.0]` rad/s

**Episode:**
- `episode_length_s = 20.0` → 1000 control steps
- `dt = 0.02 s`, `decimation = 4`

**Reset / termination:**
- Terminate when `|roll| > π/2` or `|pitch| > π/2` (robot flipped).
- Truncate at episode length (no penalty).
- Reset randomization: joint pos ±0.1 rad, base orientation ±0.1 rad, base height = 0.42 m.

**Public API (matches rsl_rl `VecEnv` contract):**
- `__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)`
- `reset() -> (obs, extras)`
- `step(actions) -> (obs, rewards, dones, infos)`
- properties: `num_obs`, `num_actions`, `num_envs`, `device`, `max_episode_length`

### 5.2 `configs/go2_config.py`

Two factory functions returning dicts/dataclasses:

- `get_cfgs()` → `(env_cfg, obs_cfg, reward_cfg, command_cfg)` — all numerical knobs above.
- `get_train_cfg(exp_name, max_iterations)` → PPO + runner config (Section 6).

### 5.3 `scripts/train.py`

```python
def main():
    args = parse_args()  # --exp_name, --num_envs, --max_iterations, --seed, --headless
    gs.init(backend=gs.gpu, logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = Go2Env(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg,
                 show_viewer=not args.headless)

    log_dir = f"logs/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
                open(f"{log_dir}/cfgs.pkl", "wb"))

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=args.max_iterations,
                 init_at_random_ep_len=True)
```

### 5.4 `scripts/eval.py`

```python
def main():
    args = parse_args()  # --exp_name, --ckpt, --num_envs=1
    gs.init(backend=gs.gpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb"))

    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg,
                 show_viewer=True)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.load(f"{log_dir}/model_{args.ckpt}.pt")
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
```

## 6. Reward Design

All rewards computed per control step. Final reward = Σ (scale × term), then `max(reward, 0)` clip (rsl_rl convention — prevents policy from learning self-termination to avoid negative reward).

| # | Term                  | Formula                                                | Scale  | Purpose                            |
|---|-----------------------|--------------------------------------------------------|--------|------------------------------------|
| 1 | `tracking_lin_vel`    | `exp(-‖cmd_xy − base_lin_vel_xy‖² / 0.25)`             | +1.0   | Match linear velocity command      |
| 2 | `tracking_ang_vel`    | `exp(-(cmd_wz − base_ang_vel_z)² / 0.25)`              | +0.5   | Match yaw rate command             |
| 3 | `lin_vel_z`           | `base_lin_vel_z²`                                      | −2.0   | Penalize bouncing                  |
| 4 | `ang_vel_xy`          | `‖base_ang_vel_xy‖²`                                   | −0.05  | Penalize roll/pitch oscillation    |
| 5 | `orientation`         | `‖projected_gravity_xy‖²`                              | −1.0   | Keep torso upright                 |
| 6 | `base_height`         | `(base_height − 0.42)²`                                | −50.0  | Maintain nominal stance height     |
| 7 | `action_rate`         | `‖aₜ − aₜ₋₁‖²`                                         | −0.005 | Smooth control                     |
| 8 | `similar_to_default`  | `Σ (qⱼ − q_defaultⱼ)²` over hip+thigh joints only      | −0.1   | Discourage weird postures          |

## 7. PPO Hyperparameters (rsl_rl `OnPolicyRunner`)

| Param                  | Value      |
|------------------------|------------|
| `num_envs`             | 4096       |
| `num_steps_per_env`    | 24         |
| `num_learning_epochs`  | 5          |
| `num_mini_batches`     | 4          |
| `learning_rate`        | 1e-3       |
| `schedule`             | `adaptive` |
| `gamma`                | 0.99       |
| `lam`                  | 0.95       |
| `entropy_coef`         | 0.01       |
| `value_loss_coef`      | 1.0        |
| `clip_param`           | 0.2        |
| `max_grad_norm`        | 1.0        |
| `desired_kl`           | 0.01       |
| `max_iterations`       | 500        |
| `save_interval`        | 50         |
| `seed`                 | 1          |

**Network:**
- Actor: MLP `[45] → [512] → [256] → [128] → [12]`, ELU activation.
- Critic: MLP `[45] → [512] → [256] → [128] → [1]`, ELU activation.
- Action distribution: Gaussian, `init_noise_std = 1.0` (learned).

**Throughput estimate:**
- 4096 envs × 24 steps = 98,304 transitions per iteration.
- ~0.5–1.0 s per iteration on RTX 5060 Ti.
- 500 iterations ≈ 5–10 minutes wall clock.

## 8. File Layout

```
StrikeRobot_genesis_development/
├── envs/
│   ├── __init__.py
│   └── go2_env.py
├── configs/
│   ├── __init__.py
│   └── go2_config.py
├── scripts/
│   ├── train.py
│   └── eval.py
├── docs/superpowers/specs/
│   └── 2026-05-12-go2-ppo-training-design.md
├── logs/                # gitignored
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

`.gitignore` additions:
```
logs/
__pycache__/
*.pyc
*.pkl
.venv/
```

## 9. Verification Plan

1. **Install check:** `python -c "import genesis; import rsl_rl"` succeeds.
2. **Smoke test:** `python scripts/train.py --exp_name smoke --num_envs 64 --max_iterations 2 --headless` finishes without error, produces `logs/smoke/model_*.pt`.
3. **Full training:** `python scripts/train.py --exp_name go2_walk --max_iterations 500 --headless`.
4. **Tensorboard inspection:** curves for `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value_function`, `Loss/surrogate`. Confirm reward trends upward and stabilizes.
5. **Success thresholds:** `mean_reward > 8.0` and `mean_episode_length > 800` after 500 iterations.
6. **Eval visual:** `python scripts/eval.py --exp_name go2_walk --ckpt 500` opens viewer, robot walks forward following command without falling.

## 10. Usage

```bash
# Install
pip install genesis-world rsl-rl-lib tensorboard

# Train (headless for speed)
python scripts/train.py --exp_name go2_walk --max_iterations 500 --headless

# Monitor
tensorboard --logdir logs/

# Playback
python scripts/eval.py --exp_name go2_walk --ckpt 500
```

## 11. Open Questions / Risks

- **Genesis API stability:** Genesis is pre-1.0; minor API differences vs. upstream `go2_train.py` example may need adjustment during implementation. Plan: pin a known-good version in `requirements.txt`.
- **rsl_rl version compatibility:** `rsl-rl-lib` 2.x changed some APIs from 1.x (notably `step()` return tuple). Plan: use 2.x and follow its `VecEnv` contract.
- **VRAM headroom:** 4096 envs should fit comfortably in 16 GB. If OOM occurs, fall back to 2048 envs (still trains fine, ~2× slower).
