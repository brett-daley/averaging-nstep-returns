from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    def __init__(self, obs_space, action_space, rng_seed: int):
        self.obs_space = obs_space
        self.action_space = action_space
        self.rng_seed = rng_seed
        self.np_random = np.random.default_rng(rng_seed)

    @abstractmethod
    def reinforce(self, obs, action, next_obs, reward, done, info={}):
        raise NotImplementedError


class ControlAgent(Agent):
    def __init__(self, obs_space, action_space, seed: int, discount: float):
        super().__init__(obs_space, action_space, seed)
        assert 0.0 <= discount <= 1.0
        self.discount = discount

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError
