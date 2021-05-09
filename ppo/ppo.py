from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from ac_policy import ActorCriticPolicy
from rollout_buffer import RolloutBuffer

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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
        load_model_path,
        save_freq,
        eval_freq,
        num_eval_episodes,
        num_eval_renders,
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
        self.load_model_path = load_model_path
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_eval_renders = num_eval_renders
        self.tensorboard = tensorboard

        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.policy = ActorCriticPolicy()
        self.rollout_buffer = RolloutBuffer(n_steps, n_envs, self.ob_dim)

        if self.load_model_path:
            self.policy = torch.load(self.load_model_path)

        if self.tensorboard:
            self.datetime = time.strftime('%m-%d-%y_%H-%M-%S')
            self.writer = SummaryWriter(log_dir='logs/{}'.format(self.datetime))

        self._obs = env.reset()
        self._next_eval = self.eval_freq
        self._next_save = self.save_freq
        self._steps = 0

    def train(self):
        self.policy.train()
        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_dataloader(self.batch_size):
                obs, actions, old_logprobs, old_values, _, _, returns, advantages = batch
                logprobs, entropy = self.policy.evaluate_actions(obs, actions)
                values = self.policy.get_values(obs)

                ratio = torch.exp(logprobs-old_logprobs)
                actor_1 = ratio*advantages
                actor_2 = torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)*advantages
                actor_loss = -torch.mean(torch.min(actor_1, actor_2))
                
                c_values = old_values + torch.clamp(values-old_values, -self.clip_coef, self.clip_coef)
                critic_1 = F.mse_loss(c_values, returns)
                critic_2 = F.mse_loss(values, returns)
                critic_loss = torch.mean(torch.max(critic_1, critic_2))

                entropy_loss = -torch.mean(entropy)

                loss = actor_loss + self.critic_coef*critic_loss + self.entropy_coef*entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)

    def learn(self):
        while self._steps < self.n_timesteps:
            self.collect_rollouts()

            al, cl, el = self.train()

            self._steps += self.n_steps*self.n_envs
            log = {
                'Mean Reward': self.rollout_buffer.mean_episode_reward,
                'Mean Length': self.rollout_buffer.mean_episode_length,
                'Actor Loss': al,
                'Critic Loss': cl,
                'Entropy Loss': el
            }
            t = PrettyTable()
            t.header = False
            t.add_row(['Timesteps', self._steps])
            for key, val in log.items():
                t.add_row([key, val])
            t.align = 'r'
            t.float_format = '.6'
            print(t)

            if self.tensorboard:
                for key, val in log.items():
                    self.writer.add_scalar(key, val, self._steps)

            if self._steps >= self._next_eval:
                mr, ml = self.evaluate_policy()
                self._next_eval += self.eval_freq
                print('Evaluating policy...')
                print('Mean Reward={}'.format(mr))
                print('Mean Length={}'.format(ml))

            if self.tensorboard and self._steps >= self._next_save:
                torch.save(self.policy, 'models/{}-{}.pt'.format(self.datetime, self._steps))
                self._next_save += self.save_freq

    def evaluate_policy(self):
        rewards = []
        lengths = []
        for e in range(self.num_eval_episodes):
            obs = self.eval_env.reset()
            episode_rewards = []
            while True:
                if e < self.num_eval_renders:
                    self.eval_env.render()
                    time.sleep(0.05)
                with torch.no_grad():
                    actions, _ = self.policy.get_actions(obs)
                    actions = actions.cpu().numpy()
                    # actions = actions.cpu().numpy()
                obs, reward, done, info = self.eval_env.step(actions)
                episode_rewards.append(reward)
                if done:
                    break
            rewards.append(np.sum(episode_rewards))
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

            self.rollout_buffer.add_transition(
                obs=self._obs, 
                actions=actions, 
                logprobs=logprobs, 
                values=values, 
                rewards=rewards, 
                dones=dones
            )

            self._obs = next_obs

        with torch.no_grad():
            values = self.policy.get_values(self._obs)
            values = values.cpu().numpy()
            self.rollout_buffer.values[-1] = values

        self.rollout_buffer.compute_advantages(self.gamma, self.gae_lambda)
        self.rollout_buffer.compute_statistics()


