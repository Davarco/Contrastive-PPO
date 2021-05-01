import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np
import gym
import argparse


device = 'cuda'


class PPO():
    def __init__(self, env, eval_env):
        self.env = env
        self.eval_env = eval_env
        self.policy = ActorCriticPolicy()
        self.rollout_buffer = RolloutBuffer(2048, (4,))

        self._obs = env.reset()

    def train(self):
        optimizer = optim.Adam(self.policy.parameters(), lr=0.005)

        # K_epochs
        for _ in range(4):
            # batch_size
            for batch in self.rollout_buffer.get_dataloader(256):
                obs, actions, next_obs, rewards, dones = batch

                values = self.policy.get_values(obs)
                next_values = self.policy.get_values(next_obs)
                # gamma
                advantages = rewards + 0.99*next_values - values

                actor_loss = -torch.mean(self.policy.forward(obs).log_prob(actions)*advantages)
                critic_loss = F.smooth_l1_loss(values, rewards + 0.99*next_values)

                optimizer.zero_grad()
                loss = actor_loss + critic_loss
                loss.backward()
                optimizer.step()

    def learn(self):
        for i in range(1000):
            self.collect_rollouts(self.env, self.policy, self.rollout_buffer)
            self.train()
            print(i)

            if i % 10 == 0:
                avg_reward = self.evaluate_policy(self.policy)
                print(avg_reward)
            
    def collect_rollouts(self, env, policy, rollout_buffer):
        self.rollout_buffer.reset()
        # buffer_size
        for i in range(2048):
            action = policy.get_action(torch.from_numpy(self._obs).to(device, torch.float32)).cpu().numpy()
            next_obs, reward, done, info = env.step(action)
            rollout_buffer.add_transition(self._obs, action, next_obs, reward, done)
            if done:
                self._obs = env.reset()

    def evaluate_policy(self, policy):
        rewards = []
        for _ in range(1):
            obs = self.eval_env.reset()
            total_reward = 0
            while True:
                self.eval_env.render()
                action = policy.get_action(torch.from_numpy(obs).to(device, torch.float32)).cpu().numpy()
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)

        return np.mean(rewards)


class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def forward(self, obs):
        logits = self.actor(obs)
        return distributions.Categorical(logits=logits)

    def get_action(self, obs):
        dis = self.forward(obs)
        return dis.sample()

    def get_values(self, obs):
        values = self.critic(obs)
        return values.squeeze()


class RolloutBuffer():
    def __init__(self, size, ob_dim):
        self.size = size
        self.ob_dim = ob_dim

    def reset(self):
        self.index = 0
        self.obs = np.zeros((self.size, *self.ob_dim), dtype=np.float32)
        self.actions = np.zeros((self.size))
        self.next_obs = np.zeros((self.size, *self.ob_dim), dtype=np.float32)
        self.rewards = np.zeros((self.size), dtype=np.float32)
        self.dones = np.zeros((self.size))

    def add_transition(self, obs, action, next_obs, reward, done):
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.next_obs[self.index] = next_obs
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index += 1
        assert self.index <= self.size, 'Buffer overflow.'

    def get_dataloader(self, batch_size):
        i = np.random.choice(np.arange(self.size), self.size, replace=False)
        j = 0
        while j < self.size:
            s = np.index_exp[j:j+batch_size]
            j += batch_size
            b = (self.obs[i][s], self.actions[i][s], self.next_obs[i][s], self.rewards[i][s], self.dones[i][s])
            yield tuple(map(lambda A: torch.from_numpy(A).to(device), b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v0', type=str)
    parser.add_argument('--procgen', action='store_true')
    args = parser.parse_args()

    if args.procgen:
        args.env_name = 'procgen:procgen-{}'.format(args.env_name)
        env = gym.make(args.env_name, distribution_mode='easy')
        eval_env = gym.make(args.env_name, distribution_mode='easy')
    else:
        env = gym.make(args.env_name)
        eval_env = gym.make(args.env_name)
    args.ob_dim = env.observation_space.shape
    args.ac_dim = env.action_space.n

    ppo = PPO(env, eval_env)
    ppo.learn()


if __name__ == '__main__':
    main()


