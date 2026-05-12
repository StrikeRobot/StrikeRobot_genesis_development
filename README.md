# StrikeRobot Genesis Development

Train a **Unitree Go2** quadruped robot to perform velocity-tracking locomotion using **PPO** in the [Genesis](https://github.com/Genesis-Embodied-AI/genesis) GPU-parallel physics engine.

The policy learns to follow velocity commands `[v_x, v_y, ω_z]` while keeping the torso upright and the gait smooth. Training runs ~4096 environments in parallel on a single GPU and converges in roughly 5–10 minutes on an RTX 5060 Ti.

## Overview

- **Robot:** Unitree Go2 (12 DoF: 4 legs × {hip, thigh, calf})
- **Task:** Velocity tracking — follow linear and angular velocity commands
- **Engine:** Genesis (GPU-accelerated, parallel envs)
- **Algorithm:** PPO via [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- **Observation:** 45-dim (base ang vel, projected gravity, commands, joint pos/vel, last actions)
- **Action:** 12-dim joint position offsets (PD-controlled at 50 Hz)

## Requirements

- Linux (tested on Ubuntu 24.04 / kernel 6.17)
- NVIDIA GPU with CUDA support (tested on RTX 5060 Ti, 16 GB VRAM)
- Python 3.10+
- PyTorch 2.x with CUDA

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd StrikeRobot_genesis_development

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pulls in `genesis-world`, `rsl-rl-lib`, and `tensorboard`. PyTorch is expected to be installed separately to match your CUDA version.

Verify the install:

```bash
python -c "import genesis, rsl_rl, torch; print('OK, CUDA:', torch.cuda.is_available())"
```

## Quick Start

### 1. Train

Headless training (fastest):

```bash
python scripts/train.py --exp_name go2_walk --max_iterations 500 --headless
```

Common flags:

| Flag                | Default     | Description                              |
|---------------------|-------------|------------------------------------------|
| `--exp_name`        | `go2_walk`  | Subfolder under `logs/` for this run     |
| `--num_envs`        | `4096`      | Parallel environments (reduce if OOM)    |
| `--max_iterations`  | `500`       | Number of PPO iterations                 |
| `--seed`            | `1`         | RNG seed                                 |
| `--headless`        | off         | Disable viewer for max training speed    |

### 2. Monitor training

In a second terminal:

```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006`. Watch for:
- `Train/mean_reward` trending up (target: `> 8.0`)
- `Train/mean_episode_length` approaching 1000 (target: `> 800`)

### 3. Play back the trained policy

```bash
python scripts/eval.py --exp_name go2_walk --ckpt 500
```

A Genesis viewer window opens and the trained Go2 walks following a velocity command.

## Project Structure

```
StrikeRobot_genesis_development/
├── envs/
│   └── go2_env.py              # Go2 environment (Genesis VecEnv)
├── configs/
│   └── go2_config.py           # env / obs / reward / command / PPO configs
├── scripts/
│   ├── train.py                # Training entry point
│   └── eval.py                 # Policy playback
├── docs/superpowers/specs/
│   └── 2026-05-12-go2-ppo-training-design.md   # Design spec
├── logs/                       # Checkpoints + Tensorboard (gitignored)
├── requirements.txt
└── README.md
```

## How It Works

Each PPO iteration:

1. 4096 parallel Go2 environments step in lockstep on the GPU.
2. The actor policy (MLP `45 → 512 → 256 → 128 → 12`, ELU) outputs joint position offsets.
3. Genesis simulates physics at 200 Hz with `decimation=4` (control at 50 Hz).
4. Rewards combine velocity tracking (positive) with penalties on bouncing, roll/pitch oscillation, body tilt, height deviation, action jerk, and joint-pose drift.
5. rsl_rl's `OnPolicyRunner` collects 24 steps × 4096 envs ≈ 98k transitions and runs 5 epochs of PPO updates with adaptive KL learning-rate scheduling.

Full hyperparameters and reward weights are documented in [the design spec](docs/superpowers/specs/2026-05-12-go2-ppo-training-design.md).

## Troubleshooting

- **CUDA OOM:** lower `--num_envs` (e.g. `2048`). Training will be ~2× slower but still converges.
- **`import genesis` fails:** ensure you used `genesis-world` (not the unrelated `genesis` PyPI package). Reinstall with `pip install --force-reinstall genesis-world`.
- **Reward not increasing past iteration 100:** check Tensorboard for `Loss/surrogate` and `Policy/mean_noise_std`. If noise collapsed early, lower `entropy_coef` or increase `init_noise_std`.
- **Viewer is laggy in eval:** that's expected with `num_envs=1` and full rendering. Training itself should always be run with `--headless`.

## References

- [Genesis physics engine](https://github.com/Genesis-Embodied-AI/genesis)
- [rsl_rl PPO implementation](https://github.com/leggedrobotics/rsl_rl)
- [Legged Gym (reward design inspiration)](https://github.com/leggedrobotics/legged_gym)
- [Unitree Go2 robot](https://www.unitree.com/go2)

## License

See [LICENSE](LICENSE).
