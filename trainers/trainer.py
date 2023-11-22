from config import Config
from samplers.sampler import Sampler
from utils.make_env import make_parallel_env
from agents.agent_factory import AgentFactory
from logger.logger import Logger
from utils.misc import get_save_checkpoint_name, get_load_checkpoint_name, get_last_timestamp
from utils.misc import n_last_eval_video_callable
import torch
from utils.seed import set_seed
import os.path as osp
import time
import numpy as np


class Trainer:
    def __init__(self, setup, root_dir):
        # instantiate Config
        self.config = Config()
        assert self.config.n_evaluation_processes == 1 and self.config.save_video, "Use serial mode for evaluation when saving videos"
        self.config.setup = setup

        self.n_envs = self.config.n_serial_envs_sampler if self.config.n_sampler_processes == 1 else self.config.n_sampler_processes

        # Set random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all
        seed = set_seed(self.config.seed)

        # instantiate training environment
        self.env = make_parallel_env(env_id=setup['train_env'],
                                     n_processes=self.config.n_sampler_processes,
                                     n_serial=self.config.n_serial_envs_sampler,
                                     max_ep_len=self.config.max_episode_len,
                                     discrete_action=setup['discrete_action'])
        self.env.seed(seed)     # set environment seed

        # instantiate and initialize agents
        self.agents = self._initialize_agents(setup)



        # instantiate Logger
        self.logger = Logger(self.config, root_dir)

        # instantiate evaluation environment
        if self.config.save_video:
            video_dict = {}
            video_save_dir = osp.join(self.logger.logdir, 'videos')
            # save n last videos per evaluation
            video_callable = n_last_eval_video_callable(n=self.config.n_vidoe_save_per_evaluation,
                                                        value=int(self.config.n_episodes_evaluation / self.config.n_serial_envs_evaluation))

            video_dict['video_save_dir'] = video_save_dir
            video_dict['video_callable'] = video_callable
        else:
            video_dict = None

        self.env_eval = make_parallel_env(env_id=setup['eval_env'],
                                          # seed=self.config.seed + 123123,
                                          n_processes=self.config.n_evaluation_processes,
                                          n_serial=self.config.n_serial_envs_evaluation,
                                          max_ep_len=self.config.max_episode_len_eval,
                                          discrete_action=setup['discrete_action'],
                                          video_dict=video_dict)
        self.env_eval.seed(seed + 123123)

        # instantiate Sampler
        self.sampler = Sampler(self.config, self.logger)
        # initialize sampler
        self.sampler.initialize(self.env, self.env_eval, self.agents)

        # load params if model loading enables
        if self.config.load_models or self.config.resume or self.config.benchmark or self.config.evaluation_mode:
            self._load()
            # osp.join(self.logger.logdir, '/../../', self.config.load_run_id, 'files/checkpoints')

        self.logger.log("Trainer initialized...")

    def train(self):
        self.logger.log("Training started...")
        itr = 0
        # run training loop
        start_time = time.time()
        collect_data_time = []
        train_mode_time = []
        optim_time = []
        after_optim_time = []


        while self.sampler.episode_completed <= self.config.n_episodes:

            # collect data by running current policy
            before_collect_data = time.time()
            self.sampler.collect_data(itr)
            collect_data_time.append(time.time() - before_collect_data)

            # get sample batch
            samples = self. sampler.sample(device=self.config.training_device)

            # train agents on sampled data
            before_train_mode = time.time()
            for agent in self.agents:
                agent.train_mode(device=self.config.training_device)
            train_mode_time.append(time.time() - before_train_mode)
            before_optim = time.time()
            for i, agent in enumerate(self.agents):
                optim_info = agent.optimize_agent(samples, self._prep_optimizer_dict(agent))
                self.logger.add_tabular(optim_info, agent=i+1)
            optim_time.append(time.time() - before_optim)
            self.logger.dump_tabular(cat_key='iteration', log=False, wandb_log=True, csv_log=False)
            self.logger.dump_tabular(cat_key='episode', csv_log=False)

            # run after optimize
            before_after_optim = time.time()
            for agent in self.agents:
                agent.after_optimize()
            after_optim_time.append(time.time() - before_after_optim)
            # save checkpoint
            if self.sampler.episode_completed % self.config.save_frequency in range(int(self.n_envs)) or \
                    self.sampler.episode_completed >= self.config.n_episodes:
                self.logger.log('Saving checkpoint at episode %d...' % self.sampler.episode_completed)
                self._save(itr=itr, episode=self.sampler.episode_completed)

            # evaluate
            if itr % self.config.evaluation_frequency == 0 and self.config.do_evaluation:
                self.sampler.evaluate()
                self.logger.dump_tabular(cat_key='evaluation', csv_log=False)


            itr += 1
            if self.sampler.episode_completed % 1000 in range(int(self.n_envs)):

                self.logger.log('Episodes: %d/%d  |  Training iteration: %d  |  Elapsed time: %f' %
                                (self.sampler.episode_completed,
                                 int(self.config.n_episodes),
                                 itr,
                                 round(time.time() - start_time, 3)))

                self.logger.log('Collect data time %f' % np.array(collect_data_time).mean(), color='gray')
                # self.logger.log('Train mode transfer time %f' % np.array(train_mode_time).mean(), color='gray')
                # self.logger.log('Optim time %f' % np.array(optim_time).mean(), color='gray')
                # self.logger.log('After optim time %f' % np.array(after_optim_time).mean(), color='gray')
                # cd_timers = self.sampler.get_timers()
                # self.logger.log('Sample mode time %f' % np.array(cd_timers['sample_time']).mean(), color='red')
                # self.logger.log('Action time %f' % np.array(cd_timers['action_time']).mean(), color='red')
                # self.logger.log('Step time %f' % np.array(cd_timers['step_time']).mean(), color='red')
                # self.logger.log('Terminal time %f' % np.array(cd_timers['terminal_time']).mean(), color='red')
                # self.logger.log('Buffer time %f' % np.array(cd_timers['buffer_time']).mean(), color='red')

                # self.sampler.reset_timers()

                collect_data_time = []
                train_mode_time = []
                optim_time = []
                after_optim_time = []

                start_time = time.time()

    # helper functions
    def _initialize_agents(self, setup):
        agent_factory = AgentFactory(self.env, self.config)
        agent_list = setup['agents']
        agent_id = 0
        agents = []
        for category in agent_list:
            num_agent_per_type, agent_type = category
            for _ in range(num_agent_per_type):
                agents.append(agent_factory(agent_id, agent_type))
                agent_id += 1
        return agents

    def _prep_optimizer_dict(self, agent):
        kwargs = dict()
        if agent.algo == 'MADDPG':
            kwargs['policies'] = [agent.policy for agent in self.agents]
            kwargs['policies_target'] = [agent.policy_target for agent in self.agents] # FIXME: fix this to work for the cases where there is no target policy
        return kwargs

    def _save(self, itr, episode):
        filename = get_save_checkpoint_name(self.logger.logdir)
        states = {}
        states['itr'] = itr
        states['episdoe'] = episode
        for agent in self.agents:
            states[agent.id] = agent.get_params()
        torch.save(states, filename)
        self.logger.log('Checkpoint saved: %s' % filename)

    def _load(self):
        path = get_load_checkpoint_name(current_root=self.logger.logdir,
                                        load_run_name=self.config.load_run_name,
                                        timestamp=self.config.load_timestamp)
        checkpoint = torch.load(path)
        for k, agent in enumerate(self.agents):
            agent.load_params(checkpoint[k])
