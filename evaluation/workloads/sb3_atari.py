#!/usr/bin/env python3
"""Stable-Baselines3 PPO on Atari Breakout with the policy on the GPU (SB3 docs "Atari Games" example, with the
rl-baselines3-zoo Atari hyperparameters: 8 envs, n_steps 128, batch 256, 4 epochs, frame stack 4).

usage: sb3_atari.py setup
       sb3_atari.py run [--steps N]          (default 100000 env steps; one final JSON line: env steps/s)

Copies: each minibatch of 256 stacked uint8 frames (256 x 4 x 84 x 84 B = 6.9 MiB) is a NumPy array moved by
th.tensor(..., device="cuda"): pageable HtoD, 16 per update (3090 trace, 5k steps: 80 such copies, 697 MiB). Measured with `cmt trace` on an A100: 6.6 GiB pageable HtoD
per 50k steps.
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appenv as c

ENV = c.ROOT / "sb3-atari"
PACKAGES = ["torch==2.14.1", "stable-baselines3==2.9.2", "gymnasium[atari]==1.4.0", "ale-py==0.12.1",
            "opencv-python-headless"]


def inner(steps):
    import ale_py
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    gym.register_envs(ale_py)
    vec_env = VecFrameStack(make_atari_env("BreakoutNoFrameskip-v4", n_envs=8, seed=0), n_stack=4)
    model = PPO("CnnPolicy", vec_env, n_steps=128, n_epochs=4, batch_size=256, learning_rate=2.5e-4,
                clip_range=0.1, vf_coef=0.5, ent_coef=0.01, verbose=0, device="cuda", seed=0)
    start = time.perf_counter()
    model.learn(total_timesteps=steps)
    seconds = time.perf_counter() - start
    c.emit("throughput", round(model.num_timesteps / seconds, 1), "env steps/s", steps=model.num_timesteps,
           learn_s=round(seconds, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "run", "inner"])
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    if args.action == "setup":
        c.venv(ENV, PACKAGES, c.torch_index())
    elif args.action == "run":
        c.exec_app([ENV / "bin" / "python", Path(__file__).resolve(), "inner", "--steps", args.steps])
    else:
        inner(args.steps)
