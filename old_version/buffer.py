from collections import namedtuple
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self._max_size = max_size
        self._storage = []

    def push(self, data):
        self._storage.append(data)
        self._storage = self._storage[-self._max_size:]

    def get_random_indices(self, batch_size):
        return np.random.randint(0, len(self._storage), min(batch_size, len(self._storage)))

    def sample_by_indices(self, indices):
        return [self._storage[i] for i in indices]

    def sample(self, batch_size):
        indices = self.get_random_indices(batch_size)
        return self.sample_by_indices(indices)