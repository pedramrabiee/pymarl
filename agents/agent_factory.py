from utils.misc import get_ac_space_info


class AgentFactory:
    def __init__(self, env, config):
        self._env = env
        self._config = config

    def __call__(self, agent_id, agent_type):
        if agent_type == 'DDPG':
            return self.instantiate_ddpg_agent(agent_id)
        if agent_type == 'MADDPG':
            return self.instantiate_maddpg_agent(agent_id)

    def agent_info_from_env(self, agent_id):
        obs_dim = self._env.observation_space[agent_id].shape[0]
        ac_dim, discrete_action = get_ac_space_info(self._env.action_space[agent_id])
        if not discrete_action:
            ac_lim_high = self._env.action_space[agent_id].high[0]  # FIXME: assume bounds are the same for all dimensions
            ac_lim_low = self._env.action_space[agent_id].low[0]  # FIXME: assume bounds are the same for all dimensions
        else:
            ac_lim_high = ac_lim_low = None
        return dict(obs_dim=obs_dim,
                    ac_dim=ac_dim,
                    discrete_action=discrete_action,
                    ac_lim_high=ac_lim_high,
                    ac_lim_low=ac_lim_low)

    def instantiate_ddpg_agent(self, agent_id):
        from agents.ddpg.ddpg import DDPGAgent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self.agent_info_from_env(agent_id)

        agent = DDPGAgent(agent_id=agent_id,
                          agent_type='DDPG',
                          obs_dim=agent_info['obs_dim'],
                          ac_dim=agent_info['ac_dim'],
                          ac_lim=dict(low=agent_info['ac_lim_low'],
                                      high=agent_info['ac_lim_high']),
                          replay_buffer=ReplayBuffer(self._config.buffer_size),
                          discrete_action=agent_info['discrete_action'])

        params = self._config.get_ddpg_params()
        agent.initialize(params)
        return agent

    def instantiate_maddpg_agent(self, agent_id):
        from agents.ddpg.maddpg import MADDPGAgent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self.agent_info_from_env(agent_id)

        agent = MADDPGAgent(agent_id=agent_id,
                            agent_type='MADDPG',
                            obs_dim=agent_info['obs_dim'],
                            ac_dim=agent_info['ac_dim'],
                            ac_lim=dict(low=agent_info['ac_lim_low'],
                                        high=agent_info['ac_lim_high']),
                            replay_buffer=ReplayBuffer(self._config.buffer_size),
                            discrete_action=agent_info['discrete_action'])
        params = self._config.get_maddpg_params()
        # sum up action and observation spaces dimension of all agents in the environment for maddpg critic
        ac_spaces = self._env.action_space
        obs_spaces = self._env.observation_space
        critic_ac_dim = critic_obs_dim = 0
        for ac_space in ac_spaces:
            ac_dim, _ = get_ac_space_info(ac_space)
            critic_ac_dim += ac_dim
        for obs_space in obs_spaces:
            critic_obs_dim += obs_space.shape[0]   # FIXME: Make a get_obs_space_shape function and incorporate visual obs space shape
        init_dict = dict(critic_obs_dim=critic_obs_dim,
                         critic_ac_dim=critic_ac_dim)
        agent.initialize(params, init_dict)
        return agent

