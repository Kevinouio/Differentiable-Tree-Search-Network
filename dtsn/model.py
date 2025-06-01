import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.model(x)


class Transition(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.action_dim = action_dim

    def forward(self, h, a):
        a_onehot = F.one_hot(a, num_classes=self.action_dim).float()
        if len(a_onehot.shape) < len(h.shape):
            a_onehot = a_onehot.unsqueeze(1).expand(-1, h.shape[1], -1)
        x = torch.cat([h, a_onehot], dim=-1)
        return self.model(x)


class Reward(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.action_dim = action_dim

    def forward(self, h, a):
        a_onehot = F.one_hot(a, num_classes=self.action_dim).float()
        if len(a_onehot.shape) < len(h.shape):
            a_onehot = a_onehot.unsqueeze(1).expand(-1, h.shape[1], -1)
        x = torch.cat([h, a_onehot], dim=-1)
        return self.model(x).squeeze(-1)


class Value(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, h):
        return self.model(h).squeeze(-1)
