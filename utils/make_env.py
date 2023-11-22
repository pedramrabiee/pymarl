import numpy as np
from utils.wrappers.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_particle_env(scenario_name, benchmark=False, discrete_action=False):
    """
    # TODO write info
    :param scenario_name:
    :param benchmark:
    :param discrete_action:
    :return:
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()

    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            scenario.benchmark_data, discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            discrete_action=discrete_action)
    return env





def make_parallel_env(env_id, n_processes, n_serial, max_ep_len, discrete_action=False, video_dict=None):
    """
    :param env_id:
    :param n_processes:
    :param n_serial:
    :param max_ep_len:
    :param discrete_action:
    :param video_dict:
    :return:
    """
    if n_processes == 1:
        return DummyVecEnv(env_fns=[lambda: make_particle_env(env_id, discrete_action=discrete_action)
                                    for _ in range(n_serial)],
                           max_ep_len=max_ep_len,
                           video_dict=video_dict)
    else:
        return SubprocVecEnv(env_fns=[lambda: make_particle_env(env_id, discrete_action=discrete_action)
                                      for _ in range(n_processes)],
                             max_ep_len=max_ep_len)

