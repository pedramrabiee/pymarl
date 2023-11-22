import torch
from torch.distributions.normal import Normal


class GaussianPD:
    def __init__(self, obs_dim):
        # generate Normal distribution callback
        pass

    def mode(self, obs):
        self.mu = self.mu_net(obs)
        return self.mu

    @property
    def dist(self):
        return Normal(self.mu, torch.exp(self.log_std))

    def log_p(self, x):
        return self.dist.log_prob(x)

    def kl(self, other):
        kl = 0.5 * ((self.dist.stddev / other.stddev) ** 2 +
                    ((self.dist.mean - other.mean) / other.stddev) ** 2 - 1 +
                    2 * torch.log(other.stddev / self.dist.stddev))  # TODO fix this
        return torch.sum(kl, dim=-1)

    def entropy(self):
        return self.dist.entropy()

    def sample(self):
        return self.dist.sample()   # TODO fix output
