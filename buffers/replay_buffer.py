import numpy as np
import torch
from utils.collections import AttrDict
from utils.misc import torchify

class ReplayBuffer:
    def __init__(self, max_size=int(100)):
        self._max_size = int(max_size)
        self._obs, self._next_obs, self._ac, self._rew, self._done = \
            None, None, None, None, None

    def push(self, experience):
        obs, ac, rew, next_obs, done = experience # FIXME: convert to namedtuple
        if self._obs is None:
            self._obs = obs[-self._max_size:]
            self._ac = ac[-self._max_size:]
            self._rew = rew[-self._max_size:]
            self._next_obs = next_obs[-self._max_size:]
            self._done = done[-self._max_size:]
        else:
            self._obs = np.concatenate([self._obs, obs])[-self._max_size:]
            self._ac = np.concatenate([self._ac, ac])[-self._max_size:]
            self._rew = np.concatenate([self._rew, rew])[-self._max_size:]
            self._next_obs = np.concatenate([self._next_obs, next_obs])[-self._max_size:]
            self._done = np.concatenate([self._done, done])[-self._max_size:]

    @property
    def buffer_size(self):
        return self._rew.shape[0]

    def get_random_indices(self, batch_size):
        return np.random.choice(np.arange(self.buffer_size),
                                size=min(batch_size, self.buffer_size),
                                replace=False)

    def sample_by_indices(self, inds, device='cpu'):
        return AttrDict(obs=torchify(self._obs[inds], device=device),
                        ac=torchify(self._ac[inds], device=device),
                        rew=torchify(self._rew[inds], device=device),
                        next_obs=torchify(self._next_obs[inds], device=device),
                        done=torchify(self._done[inds], device=device))

    def sample(self, batch_size, device='cpu'):
        """return samples as torch tensor"""
        indices = self.get_random_indices(batch_size)
        return self.sample_by_indices(indices, device=device)

    @classmethod
    def set_buffer_size(cls, max_size):
        return cls(int(max_size))