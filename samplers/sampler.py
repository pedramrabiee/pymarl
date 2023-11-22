import numpy as np
from tqdm import tqdm
import time


class Sampler:
    def __init__(self, config, logger):
        self._max_episode_len = int(config.max_episode_len)
        self._n_episode_step_per_itr = int(config.n_episode_step_per_itr)
        self._n_episodes = int(config.n_episodes)
        self._batch_size = int(config.sampling_batch_size)
        n_sampler_processes = int(config.n_sampler_processes)
        self._n_envs = n_sampler_processes if n_sampler_processes > 1 else int(config.n_serial_envs_sampler)
        self._config = config
        self.logger = logger

    def initialize(self, env, env_eval, agents):
        self.env = env
        self.agents = agents
        self.env_eval = env_eval
        self._reset_queue()
        self.logger.log('Sampler initialized...')
        self._started = False
        self._init_sample_round = True
        self._episode_counter = 0
        self._min_buffer_size = self._config.sampling_batch_size * self._config.max_episode_len
        self._n_episode_step_per_process = int(self._n_episode_step_per_itr / self._n_envs)
        # self.reset_timers()
        self._reset_buffer_queue()

    # def reset_timers(self):
    #     self.sample_mode_time = []
    #     self.get_action_time = []
    #     self.step_time = []
    #     self.terminal_analyze_time = []
    #     self.buffer_push_time = []

    # def get_timers(self):
    #     return dict(sample_time=self.sample_mode_time,
    #                 action_time=self.get_action_time,
    #                 step_time=self.step_time,
    #                 terminal_time=self.terminal_analyze_time,
    #                 buffer_time=self.buffer_push_time)

    def collect_data(self, train_itr):
        # Move policy to sampler device
        sample_dict = dict(episode=self.episode_completed)
        # before_sample_mode = time.time()
        for agent in self.agents:
            agent.sample_mode(device=self._config.sampler_device, sample_dict=sample_dict)
        # self.sample_mode_time.append(time.time() - before_sample_mode)
        # TODO: fix n_processes for the case where sampling or evaluating on gpu

        # reset the environments manually only the first time collect_data is called, the environments reset
        # automatically on done or terminal (reaching max episode length) inside the SubVec or DummyVec _worker
        if not self._started:
            obs_n = self.env.reset()
            self._started = True
            pbar = tqdm(total=self._min_buffer_size/self._n_envs, desc='Initial Sampling Progress')

        else:
            self._init_sample_round = False
            obs_n = self.last_obs

        episode_step = 0
        rollout_info = []
        while True:
            # before_get_action = time.time()
            # get actions per agent
            action_per_agent = [agent.act(obs_n[:, i]) for i, agent in enumerate(self.agents)]
            # rearrange actions to be per environment
            action_n = [[ac[i] for ac in action_per_agent] for i in range(self._n_envs)]
            # environment step
            # self.get_action_time.append(time.time() - before_get_action)

            # before_step = time.time()
            next_obs_n, rew_n, done_n, info_n = self.env.step(action_n)
            # self.step_time.append(time.time() - before_step)

            # environments wrapped with SubprocVecEnv will automatically reset any particular env when it is done or terminal
            # and save final observation in the info with the key 'terminal_observation'
            # new observation then is the observation after reset
            # before_terminal = time.time()
            terminals = ['terminal_observation' in info.keys() for info in info_n]

            if any(terminals):
                new_obs_n = next_obs_n.copy()
                for i, terminal in enumerate(terminals):
                    if terminal:
                        next_obs_n[i] = info_n[i]['terminal_observation']
                        return_per_agent, total_return = self._get_return(info_n[i]['episode_rewards'])
                        self._episode_counter += 1
                        rollout_info.append(dict(episode=self.episode_completed,
                                                 episode_return=return_per_agent,
                                                 total_return=total_return))
            # self.terminal_analyze_time.append(time.time() - before_terminal)

            self._buffer_queue(obs_n, action_n, rew_n, next_obs_n, done_n)

            # before_buffer = time.time()
            # # store transitions in replay buffer
            # for i, agent in enumerate(self.agents):
            #     agent.push_to_buffer((np.vstack(obs_n[:, i]),
            #                           action_per_agent[i],
            #                           np.vstack(rew_n[:, i]),
            #                           np.vstack(next_obs_n[:, i]),
            #                           np.vstack(done_n[:, i])))     # FIXME
            # self.buffer_push_time.append(time.time() - before_buffer)

            # push rewards and terminals to queue for logging
            self._push_to_queue(rew_n, terminals)
            # replace next_obs_n with new observation of the next episode
            if any(terminals):
                next_obs_n = new_obs_n

            obs_n = next_obs_n

            episode_step += 1
            if self._init_sample_round: pbar.update(1)

            # The first time collect_data is called, the sampling process continue until buffer_size > min_buffer_size.
            # On the following calls, loop breaks on episode_step >= n_episode_step_per_process
            if episode_step >= self._n_episode_step_per_process and self.buffer_size > self._min_buffer_size:
                # before_buffer = time.time()
                # store transitions in replay buffer

                for i, agent in enumerate(self.agents):
                    agent.push_to_buffer((np.vstack(self._obs_buf[:, i]),
                                          np.vstack(self._ac_buf[:, i]),
                                          np.vstack(self._rew_buf[:, i]),
                                          np.vstack(self._next_obs_buf[:, i]),
                                          np.vstack(self._done_buf[:, i])))     # FIXME
                self._reset_buffer_queue()
                # self.buffer_push_time.append(time.time() - before_buffer)


                if self._init_sample_round: pbar.close()

                # analyze queue and log
                logs = self._analyze_queue()
                for k, v in logs.items():
                    self.logger.add_tabular({k: v['value']}, stats=v['stats'])

                self.logger.push_tabular(rollout_info, cat_key='episode')
                # reset queue
                self._reset_queue()
                # save last observation for the next time collect_data is called
                # Notice that the environments only reset on terminal or done
                self.last_obs = obs_n
                break

    def evaluate(self):
        self.logger.log('Evaluating...')
        # Move policy to sampler device
        for agent in self.agents:
            agent.eval_mode(device=self._config.evaluation_device)

        obs_n = self.env_eval.reset()
        episode_step = eval_episode = 0
        max_episode_len_per_process = int(self._config.max_episode_len_eval / self._config.n_evaluation_processes)
        pbar = tqdm(total=max_episode_len_per_process, desc='Evaluation Progress')
        n_evaluation_processes = int(self._config.n_evaluation_processes)
        n_evaluation_envs = n_evaluation_processes if n_evaluation_processes > 1 \
            else self._config.n_serial_envs_evaluation

        eval_info = []
        while True:
            # get actions per agent
            action_per_agent = [agent.act(obs_n[:, i], explore=False) for i, agent in enumerate(self.agents)]
            # rearrange actions to be per environment
            #### FIXME fix n_processes for the serial case
            action_n = [[ac[i] for ac in action_per_agent] for i in range(n_evaluation_envs)]
            # environment step
            # TODO: you may need to rearrange actions. Check this in debug
            next_obs_n, rew_n, done_n, info_n = self.env_eval.step(action_n)

            terminals = ['terminal_observation' in info.keys() for info in info_n]
            if any(terminals):
                for i, terminal in enumerate(terminals):
                    if terminal:
                        return_per_agent, total_return = self._get_return(info_n[i]['episode_rewards'])
                        eval_info.append(dict(episode=self.episode_completed,
                                              episode_return=return_per_agent,
                                              total_return=total_return))
                        eval_episode += 1

            self._push_to_queue(rew_n, terminals)
            obs_n = next_obs_n

            episode_step += 1
            pbar.update(1)
            if eval_episode >= self._config.n_episodes_evaluation:
                # self.env_eval.reset()       # to write any unwritten videos
                pbar.close()
                # analyze queue and log
                logs = self._analyze_queue(evaluation=True)
                for k, v in logs.items():
                    self.logger.add_tabular({k: v['value']}, stats=v['stats'])
                self.logger.push_tabular(eval_info, cat_key='evaluation')
                # reset queue
                self._reset_queue()
                break

    def sample(self, device='cpu'):
        # under the assumption that all agents' buffer sizes are the same
        random_indices = self.agents[0]._buffer.get_random_indices(self._batch_size) # FIXME: _buffer
        return [agent.get_samples(random_indices, device=device) for agent in self.agents] # FIXME

    def _push_to_queue(self, rew_n, terminals):
        self.rew_queue.append(rew_n)
        self.terminal_counter += np.array(terminals, dtype=np.float32).sum()

    def _reset_queue(self):
        self.rew_queue = []
        self.terminal_counter = 0.0

    def _analyze_queue(self, evaluation=False):
        logs = {}
        rew_list = np.vstack(self.rew_queue)
        keystr = 'Evaluation/Rewards' if evaluation else 'Rewards'
        logs[keystr] = dict(value=rew_list, stats=True)
        keystr = 'Evaluation/Return' if evaluation else 'Return'
        logs[keystr] = dict(value=np.sum(rew_list, axis=0) / self.terminal_counter, stats=False)
        return logs

    @property
    def episode_completed(self):
        return self._episode_counter

    @property
    def buffer_size(self):
        return self._rew_buf.shape[0] if self._init_sample_round else self.agents[0].buffer_size + self._rew_buf.shape[0]

    def _get_return(self, rewards):
        # return per agent return and total return
        return_per_agent = np.sum(rewards, axis=0)
        total_return = np.sum(return_per_agent)
        return return_per_agent, total_return

    def _buffer_queue(self, obs, ac, rew, next_obs, done):
        """temporary replay buffer to minimize accessing to agents' buffer at each iteration of data collection"""
        if self._obs_buf is None:
            self._obs_buf = obs
            self._next_obs_buf = next_obs
            self._ac_buf = np.stack(ac)
            self._rew_buf = rew
            self._done_buf = done
        else:
            # TODO check this: obs_buf was mistakenly concatenating with next_obs
            self._obs_buf = np.concatenate((self._obs_buf, obs), axis=0)
            self._next_obs_buf = np.concatenate((self._next_obs_buf, next_obs), axis=0)
            self._ac_buf = np.concatenate((self._ac_buf, np.stack(ac)), axis=0)
            self._rew_buf = np.concatenate((self._rew_buf, rew), axis=0)
            self._done_buf = np.concatenate((self._done_buf, done), axis=0)

    def _reset_buffer_queue(self):
        self._obs_buf, self._next_obs_buf, self._ac_buf, self._rew_buf, self._done_buf = None, None, None, None, None

