import time

import cv2
import gymnasium as gym
import numpy as np

from data_structures import ImageStacker
from utils import onehot


class OneHot(gym.ObservationWrapper):
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        super().__init__(env)
        n = self.observation_space.n
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=[n], dtype=np.float32)

    def observation(self, obs):
        return onehot(obs, self.observation_space.shape, self.observation_space.dtype)


class OneHotCoordinate(gym.ObservationWrapper):
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        super().__init__(env)
        self.decode = env.unwrapped._decode
        self.dims = env.unwrapped._dims
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=[np.sum(self.dims)], dtype=np.float32)

    def observation(self, obs):
        x, y = self.decode(obs)
        dtype = self.observation_space.dtype
        X = onehot(x, self.dims[0], dtype)
        Y = onehot(y, self.dims[1], dtype)
        return np.concatenate([X, Y])


class History(gym.Wrapper):
    """Stacks the previous `history_len` observations along their last axis.
    Pads observations with zeros at the beginning of an episode."""
    def __init__(self, env, history_len=4):
        assert history_len > 1
        super().__init__(env)
        self._image_stacker = ImageStacker(history_len)

        shape = self.observation_space.shape
        self.observation_space._shape = tuple((*shape[:-1], history_len * shape[-1]))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._image_stacker.append(observation)
        stack = self._image_stacker.concatenate(axis=-1)
        return stack, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._image_stacker.append(observation, reset=True)
        stack = self._image_stacker.concatenate(axis=-1)
        return stack, info


class FireReset(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, _, _, info = self.step(1)
        return obs, info


class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, frameskip=4):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=[84, 84, 1], dtype=np.uint8)

        assert frameskip >= 1
        self.frameskip = frameskip

        # Buffer for frame pooling
        self.frame_buffer = np.zeros([2, 210, 160], dtype=self.observation_space.dtype)
        self.index = 0

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frameskip):
            _, reward, terminated, truncated, info = self.env.step(action)
            self._load_frame()
            total_reward += reward
            if terminated or truncated:
                break
        return self.observation(), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._load_frame(reset=True)
        return self.observation(), info

    def _load_frame(self, reset=False):
        if reset:
            self.frame_buffer.fill(0)
        i = self.index = 1 - self.index  # Quick way to toggle index between 0 and 1
        self.ale.getScreenGrayscale(self.frame_buffer[i])

    def observation(self):
        obs = np.max(self.frame_buffer, axis=0)  # Frame pooling
        obs = cv2.resize(obs, [84, 84], interpolation=cv2.INTER_NEAREST)
        return np.reshape(obs, [84, 84, 1])


class EnvProbe(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # These metrics are reset when an episode ends:
        self._length = None
        self._return = None
        self._done = True
        # These metrics are never reset:
        self._episodes = 0
        self._steps = 0

        # Metric histories for averaging
        self._all_lengths = []
        self._all_returns = []

        # Initial time reference point
        self._start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._length += 1
        self._return += reward

        if terminated or truncated:
            self._all_lengths.append(self._length)
            self._all_returns.append(self._return)
            self._done = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._length = 0
        self._return = 0.0
        self._done = False
        return self.env.reset(**kwargs)

    def get_total_episodes(self):
        return len(self.get_episode_returns())

    def get_total_steps(self):
        return self._steps

    def get_episode_lengths(self):
        return self._all_lengths

    def get_episode_returns(self):
        return self._all_returns

    def is_done(self):
        return self._done

    def time(self):
        return time.time() - self._start_time


class AtariRewardClipping(gym.RewardWrapper):
    """Clips rewards in {-1, 0, +1} based on their signs."""
    def reward(self, reward):
        return np.sign(reward)


class RandomNoOpReset(gym.Wrapper):
    """Perform a random number of no-ops on reset.
    The number is sampled uniformly from [`no_op_min`, `no_op_max`]."""

    def __init__(self, env, no_op_min=1, no_op_max=30):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        self.no_op_min = no_op_min
        self.no_op_max = no_op_max

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        n = self.np_random.integers(self.no_op_min, self.no_op_max, endpoint=True)
        for _ in range(n):
            observation, _, terminated, truncated, info = self.step(0)
            if terminated or truncated:
                raise RuntimeError("env terminated during no-ops")
        return observation, info


class LifeLossTermination(gym.Wrapper):
    """Signals termination when a life is lost, but only resets when the game is over."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()

        if terminated:
            self.game_over = True
            self.lives = lives
            return observation, reward, terminated, truncated, info

        if 0 < lives < self.lives:
            # We lost a life, but signal termination only if lives > 0.
            # Otherwise, the environment will handle it automatically.
            terminated, truncated = True, False

        self.lives = lives
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        assert self.game_over
        self.game_over = False
        observation, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return observation, info
