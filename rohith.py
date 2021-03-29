import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import numpy as np

device = 'cuda'

def create_models():
    policy_network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            # 16384
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 15))
    target_network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            # 16384
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 15))
    return policy_network, target_network

def train(policy_network, target_network, policy_optimizer, epochs=5, T=500, buffer, batch_size=64, gamma=0.9, switch=50):
    it = 0
    target_network.eval()
    for e in range(epochs):
        print("Epoch " + str(e))
        for t in range(T):
            experiences = buffer.sample_batch(batch_size)
            obs, action, next_obs, reward = experiences['obs'], experiences['actions'], experiences['next_obs'], experiences['reward']
            policy_network.train()
            obs = obs.to(device=device, dtype=torch.float)

            scores = policy_network(obs)
            targets = target_network(next_obs).max(1) * gamma + reward
            loss = F.cross_entropy(scores, targets)

            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()

            if t % switch == 0:
                target_network.load_state_dict(policy_network.state_dict())

def main():
    policy_network, target_network = create_models()
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=0.0001)
    buffer = ReplayBuffer(500, (64, 64, 3), 15)
    train(policy_network, target_network, policy_optimizer, epochs=5, T=500, batch_size=64)


if _name_ == '_main_':
    main()
