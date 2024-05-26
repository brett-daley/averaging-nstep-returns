from abc import ABC

import gymnasium as gym
import numpy as np

from tasks.dynamic_programming import policy_evaluation
import tasks.envs


class Task(ABC):
    def __init__(self, env_id: str, discount: float, duration: int, seed: int):
        self.env = env = tasks.envs.make(env_id)

        assert 0.0 <= discount <= 1.0
        self.discount = discount

        assert duration > 0
        self.duration = duration

        # Set random seeds
        try:
            env.seed(seed)
            self.seed = None
        except AttributeError:
            self.seed = seed
        env.action_space.seed(seed)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self.seed is not None:
            obs, info = self.env.reset(seed=self.seed)
            self.seed = None
            return obs, info

        return self.env.reset()

    def close(self):
        return self.env.close()


class PredictionTask(Task):
    def __init__(self, env_id: str, discount: float, duration: int, seed: int):
        # TODO: Pass in behavior policy as argument
        super().__init__(env_id, discount, duration, seed)
        env = self.env.unwrapped
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        n = env.action_space.n
        behavior_policy = lambda s: np.ones(n) / n
        # NOTE: This only works for gym-classics environments
        self.v_pi = policy_evaluation(env, self.discount, behavior_policy, precision=1e-9)
        self.all_observations = np.stack([self.env.observation(s) for s in env.states()])

    def policy(self):
        b_prob = t_prob = 1.0 / self.env.unwrapped.observation_space.n
        return self.env.action_space.sample(), b_prob, t_prob

    def msve(self, agent):
        v = agent.predict(self.all_observations)
        return np.mean(np.square(v - self.v_pi))


class AtariPredictionTask(Task):
    def __init__(self, game: str):
        game += "NoFrameskip-v4"
        env = gym.make(game)
        self.env = env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, noop_max=0, grayscale_newaxis=True, scale_obs=False)
        self.transition_generator = generate_atari_transitions(env)

    def step(self):
        return next(self.transition_generator)


def generate_atari_transitions(env):
    # This seed is fixed and was used by the pre-trained agents to record actions.
    # Changing it will break the benchmark.
    env.seed(1)

    with open("atari_policies/" + env.spec.id + ".txt", 'rb') as f:
        byte = f.read(1)
        episode = []

        while byte != b"":
            val = int.from_bytes(byte, 'big')

            if val == 82:
                obs, _ = env.reset()

                if episode:
                    # Set the termination flag to True for the last transition
                    last_transition = episode[-1]
                    last_transition = (*transition[:-1], True)
                    episode[-1] = last_transition

                    # Yield each transition in the episode and then clear the buffer
                    for transition in episode:
                        yield transition
                    episode.clear()

            else:
                action = val - 97
                next_obs, reward, _, _, _ = env.step(action)
                terminated = False  # Assume non-terminal until the file signals episode end
                transition = (obs, action, next_obs, reward, terminated)
                episode.append(transition)
                obs = next_obs

            byte = f.read(1)


class ControlTask(Task):
    def __init__(self, env_id: str, discount: float, duration: int, seed: int):
        super().__init__(env_id, discount, duration, seed)
        self._undisc_return = None
        self._disc_return = None
        self._time_since_done = None
        self.done = False

    @property
    def undiscounted_return(self):
        assert self.done, "must wait until end of episode"
        return self._undisc_return

    @property
    def discounted_return(self):
        assert self.done, "must wait until end of episode"
        return self._disc_return

    def step(self, action):
        assert not self.done
        obs, reward, terminated, truncated, info = super().step(action)

        self._undisc_return += reward
        self._disc_return += pow(self.discount, self._time_since_done) * reward
        self._time_since_done += 1

        self.done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def reset(self):
        self._undisc_return = 0.0
        self._disc_return = 0.0
        self._time_since_done = 0
        self.done = False
        return super().reset()
