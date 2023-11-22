import torch.optim as optim
from utils.collections import AttrDict
import torch.nn as nn

class Config:
    def __init__(self):
        self.setup = None

        # MODE
        self.resume = False
        self.benchmark = False
        self.evaluation_mode = False

        # TRAINER
        self.n_episodes = 60000                  # number of episodes
        self.n_episode_step_per_itr = 100        # number of episodes step to take before doing a training step
        self.train_iter = 25000
        self.seed = 2**32 + 475896325
        self.buffer_size = 1e6

        # TRAINING
        self.n_training_processes = 12
        self.training_device = 'cpu'

        # SAMPLER
        self.max_episode_len = 25
        self.sampling_batch_size = 1024
        self.n_sampler_processes = 8
        self.sampler_device = 'cpu'
        self.n_serial_envs_sampler = 1                  # for use in DummyVecEnv, only used when n_sampler_processes = 1

        # EVALUATION
        self.do_evaluation = True
        self.n_episodes_evaluation = 50
        self.n_vidoe_save_per_evaluation = 3
        self.max_episode_len_eval = 50
        self.n_evaluation_processes = 1
        self.evaluation_device = 'cpu'
        self.evaluation_frequency = 1000
        self.n_serial_envs_evaluation = 8                # for use in DummyVecEnv, only used when n_sampler_processes = 1
        self.save_video = True

        # LOG AND SAVE
        self.results_dir = 'results'
        self.use_wandb = True                       # Enable wandb
        self.wandb_project_name = "test_seed"       # wandb project name

        self.tensorboard = True                     # Enable TensorBoard
        self.tensorboard_update_frequency = 100     # Update TensorBoard every X training steps

        self.save_models = True                     # Enable to save models every SAVE_FREQUENCY episodes
        self.save_frequency = 10000                 # Save every SAVE_FREQUENCY episodes
        self.special_episodes_to_save = []          # Save these episode numbers, in addition to ad SAVE_FREQUENCY
        self.print_stats_frequency = 1              # Print stats every PRINT_STATS_FREQUENCY episodes
        self.stat_rolling_mean_window = 1000        # The window to average stats
        self.results_file_name = 'results.txt'      # Results filename
        self.network_name = 'network'               # Network checkpoint name
        self.add_timestamp = True                   # Add timestamp to console's

        # LOAD MODELS
        self.load_models = False
        self.load_run_name = 'run-20210117_212538-1jr59ymu'
        self.load_run_id = self.load_run_name[-8:]
        # self.load_timestamp = '20210110_131125'
        self.load_timestamp = 'last'

    ##############################
    ###### Agent params
    ##############################
    def get_ddpg_params(self):
        from explorations.ou_noise import OUNoise
        self.ddpg_params = AttrDict(
            tau=0.01,
            gamma=0.95,
            exp_strategy_cls=OUNoise,
            n_exploration_episode=self.n_episodes,
            init_noise_scale=0.3,
            final_noise_scale=0.0,
            # policy network
            pi_net_kwargs=dict(hidden_dim=64,
                               num_layers=2,
                               unit_activation=nn.ReLU,
                               out_activation=nn.Tanh,
                               batch_norm=False,
                               layer_norm=False,
                               batch_norm_first_layer=True),
            pi_optim_cls=optim.Adam,
            pi_optim_kwargs=dict(lr=1e-2,
                                 weight_decay=0),
            # grad clip
            use_clip_grad_norm=True,
            clip_max_norm=0.5,
            # q network
            q_net_kwargs=dict(hidden_dim=64,
                              num_layers=2,
                              unit_activation=nn.ReLU,
                              out_activation=nn.Identity,
                              batch_norm=False,
                              layer_norm=False,
                              batch_norm_first_layer=True),
            q_optim_cls=optim.Adam,
            q_optim_kwargs=dict(lr=1e-2,
                                weight_decay=1e-2),
        )
        return self.ddpg_params

    def get_maddpg_params(self):
        ddpg_params = self.get_ddpg_params()
        madddpg_specific_params = AttrDict()
        self.maddpg_params = AttrDict(**ddpg_params, **madddpg_specific_params)
        return self.maddpg_params
