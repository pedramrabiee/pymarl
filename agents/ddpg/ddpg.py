import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import BaseAgent
from distributions.dist_functional import one_hot_from_logits
from networks.mlp import MLPNetwork
from utils.misc import *


class DDPGAgent(BaseAgent):
    def initialize(self, params, init_dict=None):
        # instantiate policy and critic
        self.policy = MLPNetwork(in_dim=self._obs_dim, out_dim=self._ac_dim,
                                 **params.pi_net_kwargs)
        self.critic = MLPNetwork(in_dim=self._obs_dim + self._ac_dim, out_dim=1,
                                 **params.q_net_kwargs)

        # make target nets
        self.policy_target = hard_copy(self.policy)
        self.critic_target = hard_copy(self.critic)

        # instantiate optimizers
        self.policy_optimizer = params.pi_optim_cls(self.policy.parameters(), **params.pi_optim_kwargs)
        self.critic_optimizer = params.q_optim_cls(self.critic.parameters(), **params.q_optim_kwargs)

        # instantiate action exploration
        if not self._discrete_action:
            self.exploration = params.exp_strategy_cls(self._ac_dim)

        # list models
        self.models = [self.policy, self.critic, self.policy_target, self.critic_target]
        self.optimizers = [self.policy_optimizer, self.critic_optimizer]

        self.params = params

    def step(self, obs, explore=False):
        action = self.policy(obs)
        if self._discrete_action:   # discrete action
            if explore:
                action = F.gumbel_softmax(action, hard=True)
            else:
                action = one_hot_from_logits(action)
        else:   # continuous action
            if explore:
                action += torchify(self.exploration.noise())
            action = action.clamp(self._ac_lim['low'], self._ac_lim['high'])
        return action
        # TODO: WHEREVER WE TAKE QUERY FROM THE POLICY, CHANGE IT TO STEP. ALSO ADD A USE_TARGET OPTION

    def sample_mode(self, device='cpu', sample_dict=None):
        super(DDPGAgent, self).sample_mode(device=device)
        if not self._discrete_action:
            episode = sample_dict['episode']
            # set noise scale
            explr_pct_remaining = max(0, self.params.n_exploration_episode - episode) / self.params.n_exploration_episode
            scale = self.params.final_noise_scale + (self.params.init_noise_scale - self.params.final_noise_scale) * explr_pct_remaining
            self.exploration.scale = scale
            # reset noise
            self.exploration.reset()

    def optimize_agent(self, samples, optim_dict=None):
        sample = samples[self.id]
        # run one gradient descent step for Q
        self.critic_optimizer.zero_grad()
        loss_critic = self._compute_critic_loss(sample)
        loss_critic.backward()
        # clip grad
        if self.params.use_clip_grad_norm:
            clip_grad_norm_(self.critic.parameters(), self.params.clip_max_norm)
        self.critic_optimizer.step()

        # freeze q-network to save computational effort
        freeze_net(self.critic)

        # run one gradient descent for policy
        self.policy_optimizer.zero_grad()
        loss_policy = self._compute_policy_loss(sample)
        loss_policy.backward()
        # clip grad
        if self.params.use_clip_grad_norm:
            clip_grad_norm_(self.policy.parameters(), self.params.clip_max_norm)

        self.policy_optimizer.step()

        # unfreeze q-network
        unfreeze_net(self.critic)

        return {"Loss/Policy": loss_policy.cpu().data.numpy(),
                "Loss/Critic": loss_critic.cpu().data.numpy()}

    def after_optimize(self):
        # update target networks
        polyak_update(target=self.policy_target,
                      source=self.policy,
                      tau=self.params.tau)
        polyak_update(target=self.critic_target,
                      source=self.critic,
                      tau=self.params.tau)

    def _compute_critic_loss(self, sample):
        q = self.critic(torch.cat((sample.obs, sample.ac), dim=-1))

        # Bellman backup for Q fucntion
        with torch.no_grad():
            if self._discrete_action:
                target_ac = one_hot_from_logits(self.policy_target(sample.next_obs))
            else:
                target_ac = self.policy_target(sample.next_obs)

            q_pi_target = self.critic_target(torch.cat((sample.next_obs, target_ac), dim=-1))

            backup = sample.rew + self.params.gamma * (1 - sample.done) * q_pi_target

        # MSE loss against Bellman backup
        loss_critic = F.mse_loss(q, backup)
        return loss_critic

    def _compute_policy_loss(self, sample):
        if self._discrete_action:
            q_pi = self.critic(torch.cat((sample.obs, F.gumbel_softmax(self.policy(sample.obs), hard=True)), dim=-1))
        else:
            q_pi = self.critic(torch.cat((sample.obs, self.policy(sample.obs)), dim=-1))
        return -q_pi.mean()
