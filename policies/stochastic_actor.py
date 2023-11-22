from policies.base_actor import Actor
from networks.mlp import MLPNetwork
import torch
import torch.nn as nn
import numpy as np


class StochasticActor(Actor):
    def __init__(self, DistributionCls):
        self._distribution = DistributionCls() # TODO: Add Arguments


class MLPGaussianActor(StochasticActor, nn.Module):
    def __init__(self, policy_in_dim, policy_out_dim):
        super().__init__()
        # TODO: ADD EXTERNAL STD
        log_std = np.ones(policy_out_dim, dtype=np.float32)     # FIXME
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLPNetwork(policy_in_dim, policy_out_dim)     # TODO: add other network inputs from config

    def step(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return self._distribution.sample(mu, std)

    @torch.no_grad()
    def act(self, obs):
        return self.step(obs).numpy()[0]
