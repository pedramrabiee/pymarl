import numpy as np
from utils.misc import *


class BaseAgent:
    def __init__(self, agent_id, agent_type, obs_dim, ac_dim, ac_lim, replay_buffer, discrete_action=False):
        # TODO implement this
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._obs_dim = obs_dim
        self._ac_dim = ac_dim
        self._ac_lim = ac_lim
        self._buffer = replay_buffer
        self._discrete_action = discrete_action

    def initialize(self, params, init_dict=None):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs, explore=True):
        # take observation as ndarray, convert it to torch, finally return action as ndarray
        obs_torch = torch.tensor(np.vstack(obs), dtype=torch.float32)
        return self.step(obs_torch, explore=explore).numpy()

    def push_to_buffer(self, experience):
        self._buffer.push(experience)

    def get_samples(self, inds, device='cpu'):
        return self._buffer.sample_by_indices(inds, device)

    def _calculate_bootstrap_value(self, info):
        raise NotImplementedError

    def sample_mode(self, device='cpu', sample_dict=None):
        self.policy.eval()
        self.policy = to_device(self.policy, device)

    def eval_mode(self, device='cpu'):
        """Switch neural net model to evaluation mode"""
        for model in self.models:
            model.eval()
            model = to_device(model, device)

    def train_mode(self, device='cpu'):
        """Switch neural net model to training mode"""
        for model in self.models:
            model.train()
            model = to_device(model, device)

    def optimize_agent(self, samples, optim_dict=None):
        raise NotImplementedError

    def after_optimize(self):
        """implement what needs to be updated for all agents simultaneously here"""
        pass

    @property
    def id(self):
        return self._agent_id

    @property
    def algo(self):
        return self._agent_type

    def get_params(self):
        params = {}
        params['models'] = {k: model.state_dict() for k, model in enumerate(self.models)}
        params['optimizers'] = {k: optim.state_dict() for k, optim in enumerate(self.optimizers)}
        return params

    def load_params(self, checkpoint):
        for k, model in enumerate(self.models):
            model.load_state_dict(checkpoint['models'][k])
        for k, optim in enumerate(self.optimizers):
            optim.load_state_dict(checkpoint['optimizers'][k])

    @property
    def buffer_size(self):
        return self._buffer.buffer_size