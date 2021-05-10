from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from procgen import ProcgenEnv
from wrappers import ProcgenWrapper
from ac_policy import ActorCriticPolicy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import time


def evaluate_policy(env, policy, n_envs, n_steps):
    policy.eval()

    rewards = np.zeros((n_steps, n_envs), dtype=np.float64)
    dones = np.zeros((n_steps, n_envs), dtype=np.float64)
    obs = env.reset()
    for s in range(n_steps):
        with torch.no_grad():
            actions, _ = policy.get_actions(obs)
            actions = actions.cpu().numpy()
            next_obs, step_rewards, step_dones, _ = env.step(actions)
        rewards[s] = step_rewards
        dones[s] = step_dones
        obs = next_obs

        if (s+1) % 500 == 0:
            print('Finished {} steps.'.format(s+1))

    episode_rewards = []
    episode_lengths = []
    for e in range(n_envs):
        episode_reward = []
        for s in range(n_steps):
            episode_reward.append(rewards[s][e])
            if dones[s][e]:
                episode_rewards.append(np.sum(episode_reward).astype(np.float64))
                episode_lengths.append(len(episode_reward))
                episode_reward = []

    results = {
        'min_episode_reward': np.min(episode_rewards),
        'self.max_episode_reward': np.max(episode_rewards),
        'self.mean_episode_reward': np.mean(episode_rewards),
        'self.min_episode_length': np.min(episode_lengths),
        'self.max_episode_length': np.max(episode_lengths),
        'self.mean_episode_length': np.mean(episode_lengths)
    }
    return results


def main(args):
    env = ProcgenEnv(
        num_envs=args.n_envs, 
        env_name=args.env_name, 
        start_level=0, 
        num_levels=500, 
        distribution_mode='easy'
    )
    env = ProcgenWrapper(env)

    print('Environment:', args.env_name)
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    policy = torch.load(args.model_path)
    results = evaluate_policy(env, policy, args.n_envs, args.n_steps)
    
    t = PrettyTable()
    t.header = False
    for key, val in results.items():
        t.add_row([key, val])
    t.align = 'r'
    t.float_format = '.6'
    print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='fruitbot', type=str)
    parser.add_argument('--n_envs', default=64, type=int)
    parser.add_argument('--n_steps', default=5000, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    args = parser.parse_args()

    main(args)


