from torch.utils.tensorboard import SummaryWriter
from procgen import ProcgenEnv
from wrappers import ProcgenWrapper
from ppo import PPO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import time


def main(args):
    env = ProcgenEnv(
        num_envs=args.n_envs, 
        env_name=args.env_name, 
        start_level=500, 
        num_levels=50, 
        distribution_mode='easy'
    )
    eval_env = ProcgenEnv(
        num_envs=1, 
        env_name=args.env_name, 
        start_level=0, 
        num_levels=500, 
        distribution_mode='easy',
        render_mode='rgb_array'
    )
    env = ProcgenWrapper(env)
    eval_env = ProcgenWrapper(eval_env)

    print('Environment:', args.env_name)
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    ppo = PPO(
        env=env,
        n_envs=args.n_envs,
        eval_env=eval_env,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        n_timesteps=args.n_timesteps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        clip_coef=args.clip_coef,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        curl_batch_size=args.curl_batch_size,
        curl_steps=args.curl_steps,
        curl_epochs=args.curl_epochs,
        curl_lr=args.curl_lr,
        load_model_path=args.load_model_path,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        num_eval_episodes=args.num_eval_episodes,
        num_eval_renders=args.num_eval_renders,
        tensorboard=args.tensorboard
    )
    ppo.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='fruitbot', type=str)
    parser.add_argument('--n_envs', default=64, type=int)
    parser.add_argument('--gamma', default=0.999, type=float)
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--n_timesteps', default=5000000, type=int)
    parser.add_argument('--n_epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--n_steps', default=256, type=int)
    parser.add_argument('--clip_coef', default=0.2, type=float)
    parser.add_argument('--critic_coef', default=0.5, type=float)
    parser.add_argument('--entropy_coef', default=0.01, type=float)
    parser.add_argument('--curl_batch_size', default=256, type=int)
    parser.add_argument('--curl_steps', default=256, type=int)
    parser.add_argument('--curl_epochs', default=3, type=int)
    parser.add_argument('--curl_lr', default=0.0003, type=float)
    parser.add_argument('--load_model_path', default=None, type=str)
    parser.add_argument('--save_freq', default=500000, type=int)
    parser.add_argument('--eval_freq', default=500000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--num_eval_renders', default=1, type=int)
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()

    main(args)


