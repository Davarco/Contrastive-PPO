# This is not used for the actual paper, just to generate the best_test_log.csv for Gradescope.
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


def evaluate_policy(env, policy):
    policy.eval()

    rewards = np.zeros((5000, 1), dtype=np.float64)
    dones = np.zeros((5000, 1), dtype=np.float64)
    obs = env.reset()
    for s in range(5000):
        with torch.no_grad():
            actions, _ = policy.get_actions(obs)
            actions = actions.cpu().numpy()
            next_obs, step_rewards, step_dones, _ = env.step(actions)
        rewards[s] = step_rewards
        dones[s] = step_dones
        obs = next_obs
        if dones[s]:
            break

    results = {
        'reward': np.sum(rewards),
        'length': s
    }
    return results


def main(args):
    policy = torch.load(args.model_path)
    
    data = np.zeros((500, 3))
    for level in range(500):
        env = ProcgenEnv(
            num_envs=1, 
            env_name=args.env_name, 
            start_level=level, 
            num_levels=1, 
            distribution_mode='easy'
        )
        env = ProcgenWrapper(env)

        results = evaluate_policy(env, policy)
        print(level, results['reward'], results['length'])
        data[level, 0] = level
        data[level, 1] = results['reward']
        data[level, 2] = results['length']

    np.savetxt('best_test_log.csv', data, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='fruitbot', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    args = parser.parse_args()

    main(args)


