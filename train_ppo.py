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
    def __init__(
        self, 
        env, 
        eval_env,
        gamma,
        lr,
        n_timesteps,
        n_epochs,
        batch_size,
        buffer_size
    ):
        self.env = env
        self.eval_env = eval_env
        self.gamma = gamma
        self.lr = lr
        self.n_timesteps = n_timesteps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.policy = ActorCriticPolicy()
        self.rollout_buffer = RolloutBuffer(buffer_size, self.ob_dim)

        self._obs = env.reset()

    def train(self):
        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_dataloader(self.batch_size):
                obs, actions, old_logprobs, values, next_obs, rewards, dones, returns, advantages = batch

                _, logprobs = self.policy.get_action(obs)
                values = self.policy.get_values(obs)

                # advantages = returns-values
                # advantages = returns-values
                # advantages = returns
                # advantages = self.estimate_q_values(rewards.cpu().numpy(), dones.cpu().numpy())
                # advantages = returns

                # ratio = torch.exp(logprobs - old_logprobs)
                # actor_loss = -torch.mean(torch.min(ratio*advantages, torch.clamp(ratio, 0.8, 1.2)*advantages))
                actor_loss = -torch.mean(self.policy.forward(obs).log_prob(actions)*advantages)

                critic_loss = 0.5*F.mse_loss(values, returns)

                loss = actor_loss + critic_loss
                # loss = actor_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def learn(self):
        for i in range(self.n_timesteps):
            avg_reward = self.collect_rollouts(self.env, self.policy, self.rollout_buffer)
            self.train()

            print('Iteration {}: avg reward={}'.format(i, avg_reward))

            if i % 10 == 0:
                avg_reward = self.evaluate_policy(self.eval_env, self.policy)
                print(avg_reward)

    def evaluate_policy(self, env, policy):
        rewards = []
        for _ in range(1):
            obs = env.reset()
            total_reward = 0
            while True:
                env.render()
                action, _ = policy.get_action_np(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)

        return np.mean(rewards)
            
    def collect_rollouts(self, env, policy, rollout_buffer):
        rollout_buffer.reset()
        for i in range(self.buffer_size):
            action, logprob = policy.get_action_np(self._obs)
            value = policy.get_values(self._obs)
            next_obs, reward, done, info = env.step(action)
            rollout_buffer.add_transition(self._obs, action, logprob, value, next_obs, reward, done)
            if done:
                self._obs = env.reset()
            else:
                self._obs = next_obs
        rollout_buffer.compute_advantages(self.gamma)
        return rollout_buffer.get_average_reward()
    
    def estimate_q_values(self, rewards, dones):
        T = len(rewards)
        Q = []

        cumsum = 0
        for t in reversed(range(T)):
            if dones[t]:
                cumsum = rewards[t]
            else:
                cumsum = rewards[t] + self.gamma*cumsum
            Q.append(cumsum)
        
        Q = np.array(list(reversed(Q)), dtype=np.float32)
        return torch.from_numpy(Q).to(device, torch.float32)


class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()
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

    def forward(self, obs):
        logits = self.actor(obs)
        return distributions.Categorical(logits=logits)

    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
            dis = self.forward(obs)
            action = dis.sample()
            return action, dis.log_prob(action)
        else:
            dis = self.forward(obs)
            action = dis.sample()
            return action, dis.log_prob(action)

    def get_action_np(self, obs):
        action, logprob = self.get_action(obs)
        return action.cpu().numpy(), logprob.detach().cpu().numpy()

    def get_values(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
            values = self.critic(obs)
            return values.squeeze()
        else:
            values = self.critic(obs)
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
        i = np.where(self.dones == 1)[0][-1]
        return np.sum(self.rewards[:i])/np.sum(self.dones)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v0', type=str)
    parser.add_argument('--procgen', action='store_true')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_timesteps', default=1000000, type=int)
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--buffer_size', default=2048, type=int)
    args = parser.parse_args()

    if args.procgen:
        args.env_name = 'procgen:procgen-{}'.format(args.env_name)
        env = gym.make(args.env_name, distribution_mode='easy')
        eval_env = gym.make(args.env_name, distribution_mode='easy')
    else:
        env = gym.make(args.env_name)
        eval_env = gym.make(args.env_name)

    ppo = PPO(
        env=env,
        eval_env=eval_env,
        gamma=args.gamma,
        lr=args.lr,
        n_timesteps=args.n_timesteps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size
    )
    ppo.learn()


if __name__ == '__main__':
    main()


