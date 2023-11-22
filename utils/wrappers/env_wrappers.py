import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np
from utils.wrappers.monitor import Monitor


from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _worker(
        remote: mp.connection.Connection,
        parent_remote: mp.connection.Connection,
        env_fn_wrapper: CloudpickleWrapper,
        max_ep_len: int) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    ep_time = 0
    ep_rewards = []
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                ep_time += 1
                observation, reward, done, info = env.step(data)
                ep_rewards.append(reward)
                if all(done) or ep_time >= max_ep_len:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    info['episode_rewards'] = np.vstack(ep_rewards)
                    info['episode_len'] = ep_time
                    observation = env.reset()
                    ep_time = 0
                    ep_rewards = []
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                np.random.seed(data)
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                ep_time = 0
                ep_rewards = []
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self,
                 env_fns,
                 max_ep_len,
                 start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), max_ep_len)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self,
                 env_fns,
                 max_ep_len,
                 video_dict=None):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.ep_time = np.zeros(len(self.envs), dtype='int')
        self.ep_rewards = [[] for _ in range(len(self.envs))]
        self.actions = None
        self.max_ep_len = max_ep_len
        if video_dict is not None:
            self.envs[0] = Monitor(self.envs[0],
                                   video_dict['video_save_dir'],
                                   video_callable=video_dict['video_callable'],
                                   force=True,
                                   write_upon_reset=False)

    def step_async(self, actions: np.ndarray):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        # obs, rews, dones, infos = map(lambda x: np.array(x, dtype=object), zip(*results))
        obs, rews, dones, infos = map(np.array, zip(*results))

        for i in range(len(rews)):
            self.ep_rewards[i].append(rews[i])
        self.ep_time += 1

        for (i, done) in enumerate(dones):
            if all(done) or self.ep_time[i] >= self.max_ep_len:
                # save final observation where user can get it, then reset
                infos[i]['terminal_observation'] = obs[i]
                infos[i]['episode_rewards'] = np.vstack(self.ep_rewards[i])
                infos[i]['episode_len'] = self.ep_time[i]
                obs[i] = self.envs[i].reset()
                self.ep_time[i] = 0
                self.ep_rewards[i] = []

        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ep_time = np.zeros(len(self.envs), dtype='int')
        self.ep_rewards = [[] for _ in range(len(self.envs))]
        return np.array(results)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
