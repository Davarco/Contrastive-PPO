import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gym
import argparse


device = 'cuda'


class DQN():
    def __init__(self, env, params):
        self.env = env
        self.ob_dim = params['ob_dim']
        self.ac_dim = params['ac_dim']
        self.gamma = params['gamma']
        self.max_rollout_length = params['max_rollout_length']
        self.max_buffer_size = params['max_buffer_size']
        self.initial_buffer_size = params['initial_buffer_size']
        self.batch_size = params['batch_size']
        self.start_epsilon = params['start_epsilon']
        self.final_epsilon = params['final_epsilon']
        self.exploration_steps = params['exploration_steps']
        self.training_steps = params['training_steps']
        self.update_freq = params['update_freq']

        self.buffer = ReplayBuffer(self.max_buffer_size, self.ob_dim, self.ac_dim)
        self.Q = self.create_model().to(device)
        self.target_Q = self.create_model().to(device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.0001)

        self.target_Q.eval()
        self.Q.train()

        self.obs = self.env.reset()
        # self.obs = self.obs.transpose(2, 0, 1)

    def evaluate(self, num_episodes):
        rewards = []
        for _ in range(num_episodes):
            episode_reward = []
            obs = self.env.reset()
            # obs = obs.transpose(2, 0, 1)
            for _ in range(self.max_rollout_length):
                # self.env.render()
                action = self.get_action(obs, 0.05)
                obs, reward, done, _ = self.env.step(action)
                # obs = obs.transpose(2, 0, 1)
                episode_reward.append(reward)
                if done:
                    break
            rewards.append(sum(episode_reward))
        return sum(rewards)/len(rewards)

    def run_rendered_iteration(self):
        # env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy', render=True)
        # obs = env.reset()
        # obs = obs.transpose(2, 0, 1)
        # for t in range(self.max_rollout_length):
        #     action = self.get_action(obs, 0.05)
        #     obs, reward, done, info = env.step(action)
        #     obs = obs.transpose(2, 0, 1)
        #     if done:
        #         print("Episode finished after {} timesteps".format(t+1))
        #         break
        # env.close()
        pass

    def collect_transitions(self, num_transitions, eps):
        i = 0
        while i <= num_transitions:
            for _ in range(self.max_rollout_length):
                # TODO Get action from target policy
                action = self.get_action(self.obs, eps)
                next_obs, reward, done, _ = self.env.step(action)
                # next_obs = next_obs.transpose(2, 0, 1)
                transition = {
                    'obs': self.obs,
                    'action': action,
                    'next_obs': next_obs,
                    'reward': reward
                }
                self.buffer.add_transition(transition)
                self.obs = next_obs

                i += 1
                if done:
                    self.obs = self.env.reset()
                    # self.obs = self.obs.transpose(2, 0, 1)
                if i == num_transitions:
                    break

    def collect_single_transition(self, eps):
        action = self.get_action(self.obs, eps)
        next_obs, reward, done, _ = self.env.step(action)
        # next_obs = next_obs.transpose(2, 0, 1)
        transition = {
            'obs': self.obs,
            'action': action,
            'next_obs': next_obs,
            'reward': reward
        }
        self.buffer.add_transition(transition)
        self.obs = next_obs
        if done:
            self.obs = self.env.reset()
            # self.obs = self.obs.transpose(2, 0, 1)

    def get_action(self, obs, eps):
        if eps < np.random.rand():
            return self.env.action_space.sample()
        else:
            # TODO Which network?
            obs = torch.from_numpy(obs).unsqueeze(dim=0)
            obs = obs.to(device=device, dtype=torch.float)
            action = self.target_Q(obs).argmax(dim=1)
            return action.cpu().item()

    def create_model(self):
        model = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.Linear(32, 2))
        # model = nn.Sequential(
        #         nn.Conv2d(3, 16, kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #         nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #         nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #         nn.Flatten(),
        #         nn.Linear(1024, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 15))
        return model

    def train(self):
        print('Beginning training.')
        print(self.env.observation_space)
        print(self.env.action_space)
        self.collect_transitions(self.initial_buffer_size, eps=1)
        print('Finished collecting initial buffer.')

        for t in range(self.training_steps):
            eps = self.start_epsilon - (self.start_epsilon-self.final_epsilon)*t/self.exploration_steps
            eps = max(eps, self.final_epsilon)

            self.collect_single_transition(eps=eps)
            batch = self.buffer.sample_batch(self.batch_size)
            obs, actions, next_obs, rewards = batch['obs'], batch['action'], batch['next_obs'], batch['reward']

            obs = torch.from_numpy(obs)
            obs = obs.to(device=device, dtype=torch.float)
            actions = torch.from_numpy(actions)
            actions = actions.to(device=device, dtype=torch.long)
            next_obs = torch.from_numpy(next_obs)
            next_obs = next_obs.to(device=device, dtype=torch.float)
            rewards = torch.from_numpy(rewards)
            rewards = rewards.to(device=device, dtype=torch.float)

            scores = self.Q(obs).gather(1, actions.view(-1, 1)).squeeze()
            targets = self.gamma*self.target_Q(next_obs).max(dim=1)[0] + rewards
            loss = F.mse_loss(scores, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # TODO Gradient clipping, Huber loss

            if t % 1000 == 0:
                avg_score = self.evaluate(50)
                print('Iteration {}: eps={} score={}'.format(t, eps, avg_score))

            if t % self.update_freq == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())


class ReplayBuffer():
    def __init__(self, max_buffer_size, ob_dim, ac_dim):
        self.max_buffer_size = max_buffer_size
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.buffer_size = 0
        self.index = 0

        self.buffer = {
            'obs': np.zeros((max_buffer_size, *ob_dim)),
            'action': np.zeros((max_buffer_size, ac_dim)),
            'next_obs': np.zeros((max_buffer_size, *ob_dim)),
            'reward': np.zeros((max_buffer_size)),
        }

    def __len__(self):
        return self.buffer_size

    def add_transition(self, transition):
        for key in self.buffer:
            self.buffer[key][self.index] = transition[key]
            self.index = (self.index + 1) % self.max_buffer_size
            self.buffer_size = min(self.buffer_size + 1, self.max_buffer_size)
        
    def sample_batch(self, batch_size):
        b = {}
        indices = np.random.choice(np.arange(self.buffer_size), batch_size, replace=False)
        for key in self.buffer:
            b[key] = self.buffer[key][indices]
        return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='fruitbot-v0', type=str)
    parser.add_argument('--procgen', default=1, type=int)
    parser.add_argument('--ob_dim', nargs='+', default=(3, 64, 64), type=int)
    parser.add_argument('--ac_dim', default=1, type=int)
    parser.add_argument('--max_rollout_length', default=500, type=int)
    parser.add_argument('--max_buffer_size', default=100000, type=int)
    parser.add_argument('--initial_buffer_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--start_epsilon', default=1, type=float)
    parser.add_argument('--final_epsilon', default=0.05, type=float)
    parser.add_argument('--exploration_steps', default=75000, type=int)
    parser.add_argument('--training_steps', default=500000, type=int)
    parser.add_argument('--update_freq', default=10000, type=int)

    args = parser.parse_args()
    args.env_name = 'procgen:procgen-{}'.format(args.env_name) if args.procgen else args.env_name
    args.ob_dim = tuple(args.ob_dim)
    params = vars(args)

    if args.procgen:
        env = gym.make(params['env_name'], distribution_mode='easy')
    else:
        env = gym.make(params['env_name'])
    dqn = DQN(env, params)
    dqn.train()


if __name__ == '__main__':
    main()


