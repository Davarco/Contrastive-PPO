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


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.ib1 = ImpalaBlock(in_channels=3, out_channels=16)
        self.ib2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.ib3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.linear = nn.Linear(in_features=2048, out_features=256)

    def forward(self, x):
        x = self.ib1(x)
        x = self.ib2(x)
        x = self.ib3(x)
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()
        self.encoder = ImageEncoder()
        self.encoder.apply(xavier_uniform_init)
        self.actor = orthogonal_init(nn.Linear(256, 15), 0.01)
        self.critic = orthogonal_init(nn.Linear(256, 1), 1.0)

        self.encoder = self.encoder.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def forward(self, obs):
        assert 0

    def get_actions(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        logits = F.log_softmax(self.actor(self.encoder(obs)), dim=1)
        dis = distributions.Categorical(logits=logits)
        actions = dis.sample()
        return actions, dis.log_prob(actions)

    def get_values(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        values = self.critic(self.encoder(obs))
        return values.squeeze()

    def evaluate(self, obs, actions):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device, torch.float32)
        obs = self.encoder(obs)
        logits = F.log_softmax(self.actor(obs), dim=1)
        dis = distributions.Categorical(logits=logits)
        values = self.critic(obs)
        return dis.log_prob(actions), dis.entropy(), values.squeeze()


