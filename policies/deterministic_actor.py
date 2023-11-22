from policies.base_actor import Actor
from networks.mlp import MLPNetwork
import torch
import numpy as np
import torch.nn as nn


class DeterministicActor(Actor):
    def __init__(self):
        pass

class MLPDeterministicActor(nn.Module, DeterministicActor):
    def __init__(self, policy_in_dim, policy_out_dim):
        super(MLPDeterministicActor, self).__init__()
        self.mu_net = MLPNetwork(policy_in_dim, policy_out_dim)

    def step(self, obs):
        return self.mu_net(obs)

    @torch.no_grad()
    def act(self, obs):
        return self.step(obs).numpy()