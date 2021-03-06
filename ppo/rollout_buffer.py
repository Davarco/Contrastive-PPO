import torchvision.transforms as transforms
import torch
import numpy as np


device = 'cuda'


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
        self.values = np.zeros((self.n_steps+1, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

    def add_transition(self, obs, actions, logprobs, values, rewards, dones):
        self.obs[self.index] = obs
        self.actions[self.index] = actions
        self.logprobs[self.index] = logprobs
        self.values[self.index] = values
        self.rewards[self.index] = rewards
        self.dones[self.index] = dones
        self.index += 1
        assert self.index <= self.n_steps, 'Buffer overflow.'

    def get_dataloader(self, batch_size):
        buffer_size = self.n_envs*self.n_steps
        i = np.random.choice(np.arange(buffer_size), buffer_size, replace=False)
        j = 0
        while j < buffer_size:
            k = i[j:j+batch_size]
            b = (
                self.obs.reshape((buffer_size, *self.ob_dim))[k], 
                self.actions.reshape(buffer_size)[k], 
                self.logprobs.reshape(buffer_size)[k], 
                self.values[:-1].reshape(buffer_size)[k], 
                # self.rewards.reshape(buffer_size)[k], 
                # self.dones.reshape(buffer_size)[k],
                self.returns.reshape(buffer_size)[k],
                self.advantages.reshape(buffer_size)[k]
            )
            j += batch_size
            yield tuple(map(lambda A: torch.from_numpy(A).to(device), b))

    def get_curl_dataloader(self, batch_size, rotate, crop, color):
        buffer_size = self.n_envs*self.n_steps
        i = np.random.choice(np.arange(buffer_size), buffer_size, replace=False)

        # Data augmentations from the paper below
        # A Simple Framework for Contrastive Learning of Visual Representations
        # https://arxiv.org/pdf/2002.05709.pdf
        random_crop = transforms.RandomResizedCrop((64, 64), (0.5, 1.0))
        color_distort = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.8)
        ])

        j = 0
        while j < buffer_size:
            k = i[j:j+batch_size]
            j += batch_size
            anc = self.obs.reshape((buffer_size, *self.ob_dim))[k]
            pos = np.array(anc)

            if rotate:
                for r in range(4):
                    s = np.index_exp[r*(batch_size//4):(r+1)*(batch_size//4)]
                    pos[s] = np.rot90(pos[s], r, axes=(2, 3)) 

            anc, pos = tuple(map(lambda A: torch.from_numpy(A).to(device), (anc, pos))) 
            
            if crop:
                anc = random_crop(anc)
                pos = random_crop(pos)
                
            if color:
                anc = color_distort(anc)
                pos = color_distort(pos)

            yield anc, pos

    def compute_advantages(self, gamma, gae_lambda):
        self.returns = np.zeros((self.n_steps, self.n_envs), np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs), np.float32)

        # GAE implementation from the repo below
        # https://github.com/joonleesky/train-procgen-pytorch
        A = 0
        for t in reversed(range(self.n_steps)):
            delta = (self.rewards[t] + gamma*self.values[t+1]*(1-self.dones[t])) - self.values[t]
            A = gamma*gae_lambda*A*(1-self.dones[t]) + delta
            self.advantages[t] = A

        self.returns = self.advantages + self.values[:-1]
        self.advantages = (self.advantages-np.mean(self.advantages))/(np.std(self.advantages)+1e-6)

    def compute_statistics(self):
        # Ignore unfinished episodes at the end of n_steps
        episode_rewards = []
        episode_lengths = []
        for e in range(self.n_envs):
            episode_reward = []
            for s in range(self.n_steps):
                episode_reward.append(self.rewards[s][e])
                if self.dones[s][e]:
                    episode_rewards.append(np.sum(episode_reward).astype(np.float64))
                    episode_lengths.append(len(episode_reward))
                    episode_reward = []

        self.min_episode_reward = np.min(episode_rewards)
        self.max_episode_reward = np.max(episode_rewards)
        self.mean_episode_reward = np.mean(episode_rewards)
        self.min_episode_length = np.min(episode_lengths)
        self.max_episode_length = np.max(episode_lengths)
        self.mean_episode_length = np.mean(episode_lengths)


