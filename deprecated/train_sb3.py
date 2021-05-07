import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter
from procgen import ProcgenEnv
import numpy as np
import gym
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v0', type=str)
    parser.add_argument('--procgen', action='store_true')
    parser.add_argument('--gamma', default=0.999, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--n_timesteps', default=1000000, type=int)
    parser.add_argument('--n_epochs', default=8, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--buffer_size', default=256, type=int)
    parser.add_argument('--clip_coef', default=0.2, type=float)
    parser.add_argument('--critic_coef', default=0.5, type=float)
    parser.add_argument('--entropy_coef', default=0.00, type=float)
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()

    if args.procgen:
        args.env_name = 'procgen:procgen-{}'.format(args.env_name)
        # env = gym.make(args.env_name, distribution_mode='easy')
        env = make_vec_env(args.env_name, n_envs=1)
    else:
        env = gym.make(args.env_name)
        eval_env = gym.make(args.env_name)

    print('Environment:', args.env_name)
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    # eval_callback = EvalCallback(eval_env, eval_freq=10000, render=True)
    eval_callback = None

    model = PPO(
        policy='CnnPolicy', 
        env=env, 
        verbose=1, 
        learning_rate=args.lr,
        gamma=args.gamma,
        n_steps=args.buffer_size,
        batch_size=args.batch_size,
        tensorboard_log='logs' if args.tensorboard else None
    )
    model.learn(total_timesteps=args.n_timesteps, callback=eval_callback)


if __name__ == '__main__':
    main()

