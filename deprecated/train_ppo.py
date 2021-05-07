import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import argparse
import time


device = 'cuda'


class PPO():
    def __init__(
        self, 
        env, 
        eval_env,
        gamma,
        lr,
        n_timesteps,
        n_epochs,
        batch_size,
        buffer_size,
        clip_coef,
        critic_coef,
        entropy_coef,
        tensorboard
    ):
        self.env = env
        self.eval_env = eval_env
        self.gamma = gamma
        self.lr = lr
        self.n_timesteps = n_timesteps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.clip_coef = clip_coef
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.tensorboard = tensorboard

        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.policy = ActorCriticPolicy()
        self.rollout_buffer = RolloutBuffer(buffer_size, self.ob_dim)

        if self.tensorboard:
            datetime = time.strftime('%m-%d-%y_%H-%M-%S')
            self.writer = SummaryWriter(log_dir='logs/{}'.format(datetime))

        self._obs = env.reset()
        self._it = 0

    def train(self):
        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_dataloader(self.batch_size):
                obs, actions, old_logprobs, _, _, rewards, dones, returns, advantages = batch

                logprobs, entropy = self.policy.evaluate_actions(obs, actions)
                values = self.policy.get_values(obs)

                ratio = torch.exp(logprobs - old_logprobs)
                actor_loss_1 = ratio*advantages
                actor_loss_2 = torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)*advantages
                actor_loss = -torch.mean(torch.min(actor_loss_1, actor_loss_2))
                critic_loss = self.critic_coef*F.mse_loss(values, returns)
                entropy_loss = -self.entropy_coef*torch.mean(entropy)

                loss = actor_loss + critic_loss + entropy_loss

                if self.tensorboard:
                    self.writer.add_scalar('Actor Loss', actor_loss.item(), self._it)
                    self.writer.add_scalar('Critic Loss', critic_loss.item(), self._it)
                    self.writer.add_scalar('Entropy Loss', entropy_loss.item(), self._it)
                    self._it += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def learn(self):
        for i in range(self.n_timesteps):
            avg_reward = self.collect_rollouts(self.env, self.policy, self.rollout_buffer)
            self.train()

            print('Iteration {}: avg reward={}'.format(i, avg_reward))
            if self.tensorboard:
                self.writer.add_scalar('Reward', avg_reward, i)

            if (i+1) % 100 == 0:
                avg_reward = self.evaluate_policy(self.eval_env, self.policy)
                print(avg_reward)

    def evaluate_policy(self, env, policy):
        rewards = []
        for _ in range(1):
            obs = env.reset()
            total_reward = 0
            while True:
                env.render(mode='human')
                action, _ = policy.get_actions_np(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)

        return np.mean(rewards)
            
    def collect_rollouts(self, env, policy, rollout_buffer):
        rollout_buffer.reset()
        for i in range(self.buffer_size):
            action, logprob = policy.get_actions_np(self._obs)
            value = policy.get_values(self._obs)
            next_obs, reward, done, info = env.step(action)
            rollout_buffer.add_transition(self._obs, action, logprob, value, next_obs, reward, done)
            if done:
                self._obs = env.reset()
            else:
                self._obs = next_obs
        rollout_buffer.compute_advantages(self.gamma)
        return rollout_buffer.get_average_reward()
    

class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()
        self.features = nn.Sequential(
        )
        self.actor = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        # self.actor = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 15)
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )
        self.features = self.features.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def forward(self, obs):
        logits = self.actor(self.features(obs))
        return distributions.Categorical(logits=logits)

    def process_obs(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(dim=0)
        if len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)/255
        # Norm?
        return obs

    def get_actions(self, obs):
        obs = self.process_obs(obs)
        dis = self.forward(obs)
        action = dis.sample()
        return action.squeeze(), dis.log_prob(action)

    def get_actions_np(self, obs):
        action, logprob = self.get_actions(obs)
        return action.cpu().numpy(), logprob.detach().cpu().numpy()

    def evaluate_actions(self, obs, actions):
        obs = self.process_obs(obs)
        dis = self.forward(obs)
        return dis.log_prob(actions), dis.entropy()

    def get_values(self, obs):
        obs = self.process_obs(obs)
        values = self.critic(self.features(obs))
        return values.squeeze()


class RolloutBuffer():
    def __init__(self, size, ob_dim):
        self.size = size
        self.ob_dim = ob_dim

    def reset(self):
        self.index = 0
        self.obs = np.zeros((self.size, *self.ob_dim), dtype=np.float32)
        self.actions = np.zeros(self.size)
        self.logprobs = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.next_obs = np.zeros((self.size, *self.ob_dim), dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)

    def add_transition(self, obs, action, logprob, value, next_obs, reward, done):
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.logprobs[self.index] = logprob
        self.values[self.index] = value
        self.next_obs[self.index] = next_obs
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index += 1
        assert self.index <= self.size, 'Buffer overflow.'

    def get_dataloader(self, batch_size):
        # i = np.random.choice(np.arange(self.size), self.size, replace=False)
        i = np.arange(self.size)
        j = 0
        while j < self.size:
            s = np.index_exp[j:j+batch_size]
            j += batch_size
            b = (
                self.obs[i][s], 
                self.actions[i][s], 
                self.logprobs[i][s], 
                self.values[i][s], 
                self.next_obs[i][s], 
                self.rewards[i][s], 
                self.dones[i][s],
                self.returns[i][s],
                self.advantages[i][s]
            )
            yield tuple(map(lambda A: torch.from_numpy(A).to(device), b))

    def get_average_reward(self):
        i = np.where(self.dones == 1)[0]
        if len(i) == 0:
            return np.sum(self.rewards)
        else:
            end = i[-1]
            return np.sum(self.rewards[:end])/np.sum(self.dones)

    def compute_advantages(self, gamma):
        T = len(self.rewards)
        self.returns = np.zeros(T, np.float32)

        cumsum = 0
        for t in reversed(range(T)):
            if self.dones[t]:
                cumsum = self.rewards[t]
            else:
                cumsum = self.rewards[t] + gamma*cumsum
            self.returns[t] = cumsum
        
        self.returns = (self.returns-np.mean(self.returns))/(np.std(self.returns)+1e-6)
        self.advantages = self.returns - self.values


class ImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v0', type=str)
    parser.add_argument('--procgen', action='store_true')
    parser.add_argument('--gamma', default=0.995, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--n_timesteps', default=1000000, type=int)
    parser.add_argument('--n_epochs', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--buffer_size', default=512, type=int)
    parser.add_argument('--clip_coef', default=0.2, type=float)
    parser.add_argument('--critic_coef', default=0.5, type=float)
    parser.add_argument('--entropy_coef', default=0.00, type=float)
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()

    if args.procgen:
        args.env_name = 'procgen:procgen-{}'.format(args.env_name)
        env = gym.make(args.env_name, distribution_mode='easy')
        eval_env = gym.make(args.env_name, distribution_mode='easy', render=True)
    else:
        env = gym.make(args.env_name)
        eval_env = gym.make(args.env_name)

    # if len(env.observation_space.shape) > 1:
    #     env = ImageWrapper(env)

    print('Environment:', args.env_name)
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)

    ppo = PPO(
        env=env,
        eval_env=eval_env,
        gamma=args.gamma,
        lr=args.lr,
        n_timesteps=args.n_timesteps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        clip_coef=args.clip_coef,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        tensorboard=args.tensorboard
    )
    ppo.learn()


if __name__ == '__main__':
    main()


