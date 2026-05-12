"""Config factory for Go2 velocity-tracking PPO training (rsl_rl 5.x schema)."""


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
        "termination_if_roll_greater_than": 1.5708,
        "termination_if_pitch_greater_than": 1.5708,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
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


def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    return {
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
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
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
                "learn_std": True,
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 50,
        "experiment_name": exp_name,
        "run_name": "",
        "seed": 1,
        "max_iterations": max_iterations,
    }
