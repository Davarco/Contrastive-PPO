from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from ac_policy import ImageEncoder, ActorCriticPolicy
from rollout_buffer import RolloutBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
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
        curl_batch_size,
        curl_steps,
        curl_epochs,
        curl_lr,
        curl_lr_lin_sch,
        curl_rotate,
        curl_crop,
        curl_distort,
        model_path,
        save_freq,
        eval_freq,
        num_eval_episodes,
        num_eval_renders,
        tensorboard,
        run_name
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
        self.curl_batch_size = curl_batch_size
        self.curl_steps = curl_steps
        self.curl_epochs = curl_epochs
        self.curl_lr = curl_lr
        self.curl_lr_lin_sch = curl_lr_lin_sch
        self.curl_rotate = curl_rotate
        self.curl_crop = curl_crop
        self.curl_distort = curl_distort
        self.model_path = model_path
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_eval_renders = num_eval_renders
        self.tensorboard = tensorboard
        self.run_name = run_name
        
        if not run_name:
            self.run_name = time.strftime('%m-%d-%y_%H-%M-%S')
        
        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.policy = ActorCriticPolicy()
        self.rollout_buffer = RolloutBuffer(n_steps, n_envs, self.ob_dim)
        self.curl_rollout_buffer = RolloutBuffer(self.curl_steps, n_envs, self.ob_dim)

        if self.model_path:
            self.policy = torch.load(self.model_path)

        # self.key_encoder = ImageEncoder().to(device)
        self.W = nn.Parameter(torch.randn(256, 256, requires_grad=True, device=device))

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir='logs/{}'.format(self.run_name))

        self._obs = env.reset()
        self._next_eval = self.eval_freq
        self._next_save = self.save_freq
        self._steps = 0

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.curl_optimizer = optim.Adam(list(self.policy.encoder.parameters()) + [self.W], lr=self.curl_lr)

    def train(self):
        self.policy.train()

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get_dataloader(self.batch_size):
                # obs, actions, old_logprobs, old_values, _, _, returns, advantages = batch
                obs, actions, old_logprobs, old_values, returns, advantages = batch
                logprobs, entropy, values = self.policy.evaluate(obs, actions)

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

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        return tuple(map(np.mean, [actor_losses, critic_losses, entropy_losses]))

    def train_curl(self):
        self.policy.train()

        i = 0
        for e in range(self.curl_epochs):
            if self.curl_lr_lin_sch:
                for g in self.curl_optimizer.param_groups:
                    g['lr'] = (1 - e/self.curl_epochs)*self.curl_lr

            curl_losses = []
            curl_scores = []

            for batch in self.curl_rollout_buffer.get_curl_dataloader(
                batch_size=self.curl_batch_size, 
                rotate=self.curl_rotate, 
                crop=self.curl_crop, 
                color=self.curl_distort
            ):
                # Pseudocode from the paper below
                # CURL: Contrastive Unsupervised Representations for Reinforcement Learning
                # https://arxiv.org/pdf/2004.04136.pdf
                anc, pos = batch
                anc_enc = self.policy.encoder(anc)
                pos_enc = self.policy.encoder(pos)
                logits = torch.matmul(anc_enc, torch.matmul(self.W, torch.transpose(pos_enc, 0, 1)))
                values, indices = torch.max(logits, dim=1)
                logits = logits - values
                labels = torch.arange(logits.shape[0]).to(device)
                curl_loss = F.cross_entropy(logits, labels)
                curl_score = torch.sum(labels == indices)/logits.shape[0]

                self.curl_optimizer.zero_grad()
                curl_loss.backward()
                self.curl_optimizer.step()

                curl_losses.append(curl_loss.item())
                curl_scores.append(curl_score.item())

                i += 1
                if i % 50 == 0:
                    curl_loss = np.mean(curl_losses)
                    curl_score = np.mean(curl_scores)
                    curl_losses = []
                    curl_scores = []

                    if self.tensorboard:
                        self.writer.add_scalar('Contrastive Loss', curl_loss.item(), i)
                        self.writer.add_scalar('Contrastive Score', curl_score.item(), i)

                    log = {
                        'Contrastive Loss': curl_loss,
                        'Contrastive Score': curl_score
                    }
                    t = PrettyTable()
                    t.header = False
                    t.add_row(['Iteration', i])
                    t.add_row(['Epoch', e])
                    t.add_row(['Contrastive Loss', curl_loss])
                    t.add_row(['Contrastive Score', curl_score])
                    t.align = 'r'
                    t.float_format = '.6'
                    print(t)

    def learn(self):
        if self.curl_steps > 0:
            self.collect_rollouts(self.curl_rollout_buffer, self.curl_steps)
            print('Finished collecting pretraining rollouts.')
            self.train_curl()
            self._steps += self.curl_steps*self.n_envs

        while self._steps < self.n_timesteps:
            self.collect_rollouts(self.rollout_buffer, self.n_steps)

            actor_loss, critic_loss, entropy_loss = self.train()

            self._steps += self.n_steps*self.n_envs
            log = {
                'Mean Reward': self.rollout_buffer.mean_episode_reward,
                'Mean Length': self.rollout_buffer.mean_episode_length,
                'Actor Loss': actor_loss,
                'Critic Loss': critic_loss,
                'Entropy Loss': entropy_loss
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
                torch.save(self.policy, 'models/checkpoints/{}-{}.pt'.format(self.run_name, self._steps))
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
            
    def collect_rollouts(self, rollout_buffer, steps):
        self.policy.eval()
        rollout_buffer.reset()
        for _ in range(steps):
            with torch.no_grad():
                actions, logprobs = self.policy.get_actions(self._obs)
                values = self.policy.get_values(self._obs)
                actions = actions.cpu().numpy()
                logprobs = logprobs.cpu().numpy()
                values = values.cpu().numpy()

                next_obs, rewards, dones, _ = self.env.step(actions)

            rollout_buffer.add_transition(
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
            rollout_buffer.values[-1] = values

        rollout_buffer.compute_advantages(self.gamma, self.gae_lambda)
        rollout_buffer.compute_statistics()


