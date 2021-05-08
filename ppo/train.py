from torch.utils.tensorboard import SummaryWriter
from procgen import ProcgenEnv
from wrappers import ProcgenWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import time


device = 'cuda'


class PPO():
    def __init__(
        self, 
        env, 
        n_envs,
        eval_env,
        gamma,
        gae_lambda,
        lr,
        n_timesteps,
        n_epochs,
        batch_size,
        n_steps,
        clip_coef,
        critic_coef,
        entropy_coef,
        tensorboard
    ):
        self.env = env
        self.n_envs = n_envs
        self.eval_env = eval_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.n_timesteps = n_timesteps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.clip_coef = clip_coef
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.tensorboard = tensorboard

        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.policy = ActorCriticPolicy()
        self.rollout_buffer = RolloutBuffer(n_steps, n_envs, self.ob_dim)

        if self.tensorboard:
            datetime = time.strftime('%m-%d-%y_%H-%M-%S')
            self.writer = SummaryWriter(log_dir='logs/{}'.format(datetime))

        self._obs = env.reset()
        self._steps = 0
        self._it = 0

    def train(self):
        self.policy.train()
        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        actor_losses = []
        critic_losses = []

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_dataloader(self.batch_size):
                obs, actions, old_logprobs, old_values, _, _, _, _, returns, advantages = batch
                logprobs, entropy = self.policy.evaluate_actions(obs, actions)
                values = self.policy.get_values(obs)

                ratio = torch.exp(logprobs - old_logprobs)
                actor_loss_1 = ratio*advantages
                actor_loss_2 = torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)*advantages
                actor_loss = -torch.mean(torch.min(actor_loss_1, actor_loss_2))
                
                critic_loss = self.critic_coef*F.mse_loss(values, returns)
                
                entropy_loss = -self.entropy_coef*torch.mean(entropy)

                loss = actor_loss + critic_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    def learn(self):
        while self._steps < self.n_timesteps:
            self.collect_rollouts()

            al, cl = self.train()

            self._steps += self.n_steps*self.n_envs
            print()
            print('Iteration={}'.format(self._it))
            print('Timesteps={}'.format(self._steps))
            print('Mean Reward={}'.format(self.rollout_buffer.mean_episode_reward))
            print('Min Reward={}'.format(self.rollout_buffer.min_episode_reward))
            print('Max Reward={}'.format(self.rollout_buffer.max_episode_reward))
            print('Mean Length={}'.format(self.rollout_buffer.mean_episode_length))
            print('Actor Loss={}'.format(al))
            print('Critic Loss={}'.format(cl))

            # if self.tensorboard:
            #     self.writer.add_scalar('Reward', avg_reward, i)

            self._it += 1
            if self._it % 10 == 0:
                mr, ml = self.evaluate_policy()
                print('Evaluating policy...')
                print('Mean Reward={}'.format(mr))
                print('Mean Length={}'.format(ml))

    def evaluate_policy(self):
        rewards = []
        lengths = []
        for _ in range(1):
            obs = self.eval_env.reset()
            episode_rewards = []
            while True:
                self.eval_env.render()
                with torch.no_grad():
                    actions, _ = self.policy.get_actions(obs)
                    actions = actions.unsqueeze(dim=0).cpu().numpy()
                    # actions = actions.cpu().numpy()
                obs, reward, done, info = self.eval_env.step(actions)
                episode_rewards.append(reward)
                if done:
                    break
                else:
                    time.sleep(0.02)
            rewards.append(sum(episode_rewards))
            lengths.append(len(episode_rewards))

        return np.mean(rewards), np.mean(lengths)
            
    def collect_rollouts(self):
        self.policy.eval()
        self.rollout_buffer.reset()
        for _ in range(self.n_steps):
            with torch.no_grad():
                actions, logprobs = self.policy.get_actions(self._obs)
                values = self.policy.get_values(self._obs)
                actions = actions.cpu().numpy()
                logprobs = logprobs.cpu().numpy()
                values = values.cpu().numpy()

                next_obs, rewards, dones, _ = self.env.step(actions)
                next_values = self.policy.get_values(next_obs)
                next_values = next_values.cpu().numpy()

            self.rollout_buffer.add_transition(
                obs=self._obs, 
                actions=actions, 
                logprobs=logprobs, 
                values=values, 
                next_obs=next_obs, 
                next_values=next_values,
                rewards=rewards, 
                dones=dones
            )

            # if dones:
            #     self._obs = self.env.reset()
            # else:
            self._obs = next_obs

        self.rollout_buffer.compute_advantages(self.gamma, self.gae_lambda)
        self.rollout_buffer.compute_statistics()
    

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU()
        # )
        self.features = nn.Sequential(
            ImpalaBlock(in_channels=3, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*8*8, out_features=256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(256, 15)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 1)
        )

        xavier_uniform_init(self.features)
        orthogonal_init(self.actor, 0.01)
        orthogonal_init(self.critic, 1.0)

        self.features = self.features.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def forward(self, obs):
        logits = F.log_softmax(self.actor(self.features(obs)), dim=1)
        return distributions.Categorical(logits=logits)

    def get_actions(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        dis = self.forward(obs)
        actions = dis.sample()
        return actions.squeeze(), dis.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        dis = self.forward(obs)
        return dis.log_prob(actions), dis.entropy()

    def get_values(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        values = self.critic(self.features(obs))
        return values.squeeze()


class RolloutBuffer():
    def __init__(self, n_steps, n_envs, ob_dim):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ob_dim = ob_dim

    def reset(self):
        self.index = 0
        self.obs = np.zeros((self.n_steps, self.n_envs, *self.ob_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs))
        self.logprobs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.next_obs = np.zeros((self.n_steps, self.n_envs, *self.ob_dim), dtype=np.float32)
        self.next_values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

    def add_transition(self, obs, actions, logprobs, values, next_obs, next_values, rewards, dones):
        self.obs[self.index] = obs
        self.actions[self.index] = actions
        self.logprobs[self.index] = logprobs
        self.values[self.index] = values
        self.next_obs[self.index] = next_obs
        self.next_values[self.index] = next_values
        self.rewards[self.index] = rewards
        self.dones[self.index] = dones
        self.index += 1
        assert self.index <= self.n_steps, 'Buffer overflow.'

    def get_dataloader(self, batch_size):
        buffer_size = self.n_envs*self.n_steps
        i = np.random.choice(np.arange(buffer_size), buffer_size, replace=False)
        # i = np.arange(buffer_size)
        j = 0
        while j < buffer_size:
            s = np.index_exp[j:j+batch_size]
            j += batch_size
            b = (
                self.obs.reshape((buffer_size, *self.ob_dim))[i][s], 
                self.actions.reshape(buffer_size)[i][s], 
                self.logprobs.reshape(buffer_size)[i][s], 
                self.values.reshape(buffer_size)[i][s], 
                self.next_obs.reshape((buffer_size, *self.ob_dim))[i][s], 
                self.next_values.reshape(buffer_size)[i][s], 
                self.rewards.reshape(buffer_size)[i][s], 
                self.dones.reshape(buffer_size)[i][s],
                self.returns.reshape(buffer_size)[i][s],
                self.advantages.reshape(buffer_size)[i][s]
            )
            yield tuple(map(lambda A: torch.from_numpy(A).to(device), b))

    def compute_advantages(self, gamma, gae_lambda):
        self.returns = np.zeros((self.n_steps, self.n_envs), np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs), np.float32)

        A = 0
        for t in reversed(range(self.n_steps)):
            delta = (self.rewards[t] + gamma*self.next_values[t]*(1-self.dones[t])) - self.values[t]
            A = gamma*gae_lambda*A*(1-self.dones[t]) + delta
            self.advantages[t] = A
        
        self.returns = self.advantages + self.values

        self.advantages = (self.advantages-np.mean(self.advantages))/(np.std(self.advantages)+1e-6)

    def compute_statistics(self):
        episode_rewards = []
        episode_lengths = []
        for e in range(self.n_envs):
            episode_reward = []
            for s in range(self.n_steps):
                episode_reward.append(self.rewards[s][e])
                if self.dones[s][e]:
                    episode_reward.clear()
            episode_rewards.append(sum(episode_reward))
            episode_lengths.append(len(episode_reward))

        self.min_episode_reward = np.min(episode_rewards)
        self.max_episode_reward = np.max(episode_rewards)
        self.mean_episode_reward = np.mean(episode_rewards)
        self.min_episode_length = np.min(episode_lengths)
        self.max_episode_length = np.max(episode_lengths)
        self.mean_episode_length = np.mean(episode_lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v0', type=str)
    parser.add_argument('--n_envs', default=64, type=int)
    parser.add_argument('--procgen', action='store_true')
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
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()

    if args.procgen:
        env = ProcgenEnv(
            num_envs=args.n_envs, 
            env_name=args.env_name, 
            start_level=0, 
            num_levels=0, 
            distribution_mode='easy'
        )
        eval_env = ProcgenEnv(
            num_envs=1, 
            env_name=args.env_name, 
            start_level=0, 
            num_levels=0, 
            distribution_mode='easy',
            render_mode='rgb_array'
        )
        env = ProcgenWrapper(env)
        eval_env = ProcgenWrapper(eval_env)
    else:
        env = gym.make(args.env_name)
        eval_env = gym.make(args.env_name)

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
        tensorboard=args.tensorboard
    )
    ppo.learn()


if __name__ == '__main__':
    main()


