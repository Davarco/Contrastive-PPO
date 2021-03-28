import numpy as np
import gym
import cv2
import argparse
import os


def collect_observations(num_obs, env_name, max_rollout_length=200, procgen=True, render=False, data_dir='data'):
    if procgen:
        env = gym.make('procgen:procgen-{}'.format(env_name), distribution_mode='easy')
        print(env.observation_space)
        print(env.action_space)
    else:
        env = gym.make(env_name)
    i = 0
    os.makedirs('{}/{}'.format(data_dir, env_name), exist_ok=True)
    while i != num_obs:
        obs = env.reset()
        for _ in range(max_rollout_length):
            action = env.action_space.sample()
            cv2.imwrite('{}/{}/{}.jpg'.format(data_dir, env_name, i), obs)
            obs, reward, done, _ = env.step(action)
            i += 1
            if i == num_obs or done:
                break

    env.close()
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_obs', type=int)
    parser.add_argument('env_name', type=str)
    parser.add_argument('--render', default=False, type=bool)

    args = parser.parse_args()
    args = vars(args)

    num_obs = args['num_obs']
    env_name = args['env_name']
    render = args['render']
 
    collect_observations(num_obs, env_name, render=render)


if __name__ == '__main__':
    main()
