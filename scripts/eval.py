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
                   help="0 = run forever; otherwise stop after N steps.")
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
