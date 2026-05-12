"""Train Go2 velocity-tracking policy with PPO (rsl_rl) on Genesis."""
import argparse
import os
import pickle
import sys

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
