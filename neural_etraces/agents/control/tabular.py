import gymnasium as gym
import numpy as np

from agents import ControlAgent
from agents.control.dqn import argmax
import schedules
from utils import egreedy


class Sarsa(ControlAgent):
    def __init__(self, observation_space, action_space, seed, discount,
                 alpha=0.01, lambd=0.0, epsilon=0.05):
        assert isinstance(observation_space, gym.spaces.Discrete)
        assert isinstance(action_space, gym.spaces.Discrete)

        super().__init__(observation_space, action_space, seed, discount)
        self.alpha = alpha
        self.lambd = lambd
        self.epsilon_schedule = schedules.make(epsilon)

        S, A = observation_space.n, action_space.n
        self.q = np.zeros([S, A])
        self.z = self.q.copy()
        self.t = 0
        self.next = None  # Next (action, probability)

    def act(self, state):
        self.t += 1
        if self.next is not None:
            return self.next
        return self._sample_action(state)

    def _sample_action(self, state):
        epsilon = self.epsilon_schedule(self.t)
        assert 0.0 <= epsilon <= 1.0

        if self.np_random.random() <= epsilon:
            prob = epsilon / self.action_space.n
            return self.action_space.sample(), prob

        prob = 1 - epsilon + (epsilon / self.action_space.n)
        return argmax(self.q[state]), prob

    def distr(self, Q):
        epsilon = self.epsilon_schedule(self.t)
        return egreedy(Q, epsilon)

    def reinforce(self, s, a, sp, r, terminated, truncated, b_prob):
        td_error = r - self.q[s,a]
        if not terminated:
            self.next = self._sample_action(sp)
            ap = self.next[0]
            td_error += self.discount * self.q[sp,ap]
        else:
            self.next = None

        self.z *= self.discount * self.lambd
        self.z[s,a] += 1.0

        self.q += self.alpha * td_error * self.z

        if terminated or truncated:
            self.z *= 0.0

    def v(self, s):
        q = self.q[s]
        t_probs = np.array(self.distr(q[None]))
        return np.sum(t_probs * q)


class QPi(Sarsa):
    def reinforce(self, s, a, sp, r, terminated, truncated, b_prob):
        td_error = r - self.q[s,a]
        if not terminated:
            td_error += self.discount * self.v(sp)

        self.z *= self.discount * self.lambd
        self.z[s,a] += 1.0

        self.q += self.alpha * td_error * self.z

        if terminated or truncated:
            self.z *= 0.0


class ESarsa(Sarsa):
    def reinforce(self, s, a, sp, r, terminated, truncated, b_prob):
        v = self.v(s)
        td_error = r - v
        if not terminated:
            td_error += self.discount * self.v(sp)

        self.z *= self.discount * self.lambd

        step = td_error * self.z
        step[s,a] += v - self.q[s,a] + td_error
        self.q += self.alpha * step

        self.z[s,a] += 1.0
        if terminated or truncated:
            self.z *= 0.0


class IESarsa(Sarsa):
    def reinforce(self, s, a, sp, r, terminated, truncated, b_prob):
        v = self.v(s)
        td_error = r - v
        if not terminated:
            self.next = self._sample_action(sp)
            ap = self.next[0]
            td_error += self.discount * self.q[sp,ap]
        else:
            self.next = None

        self.z *= self.discount * self.lambd

        step = td_error * self.z
        step[s,a] += v - self.q[s,a] + td_error
        self.q += self.alpha * step

        self.z[s,a] += 1.0
        if terminated or truncated:
            self.z *= 0.0
