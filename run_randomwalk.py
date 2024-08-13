import argparse
import os

import gymnasium as gym
import gym_classics
gym_classics.register('gymnasium')
import numpy as np

from averaging_nstep_returns import cmdline, tasks
from averaging_nstep_returns.utils import jax_device_context
import averaging_nstep_returns.returns.numpy.V as returns
from averaging_nstep_returns.tasks.dynamic_programming import policy_evaluation


def main(**kwargs):  # Hook for automation
    kwargs = cmdline.insert_defaults(kwargs)
    cmdline.assert_not_none(kwargs)

    with jax_device_context(force_cpu=True):
        return run(**kwargs)


def run(estimator: str, alpha: float, seed: int, verbose: bool = False):
    assert 0.0 < alpha <= 1.0
    discount = 1.0

    def calculate_returns(V_table, states, next_states, rewards, dones):
        v = V_table[states]
        next_v = V_table[next_states]
        terminateds = dones
        truncateds = np.zeros_like(dones)
        return returns.calculate(estimator, v, next_v, rewards, terminateds, truncateds, discount)

    def offline_update(V, states, next_states, rewards, dones):
        Gs = calculate_returns(V, states, next_states, rewards, dones)
        T = len(Gs)
        for t in range(T):
            s = states[t]
            V[s] += alpha * (Gs[t] - V[s])
        return V

    env_id = '19Walk-v0'
    env = gym.make(env_id)
    env.action_space.seed(seed)

    if not os.path.exists('walk.npy'):
        n = env.action_space.n
        behavior_policy = lambda s: np.ones(n) / n
        # NOTE: This only works for gym-classics environments
        V_pi = policy_evaluation(env, discount, behavior_policy, precision=1e-9)
        np.save('walk', V_pi)
    else:
        V_pi = np.load('walk.npy')

    V = np.zeros(env.observation_space.n)

    def rms(V):
        return np.sqrt(np.mean(np.square(V - V_pi)))

    rms_errors = []
    for i in range(10):  # For each episode
        transitions = []
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            assert not truncated
            transitions.append((state, next_state, reward, done))
            state = next_state

        batch = tuple(map(np.array, zip(*transitions)))
        V = offline_update(V, *batch)

        rms_errors.append(rms(V))
        if verbose:
            print(V)
            print(i + 1, rms_errors[-1])
            print()

    # print(rms_errors)
    return np.array(rms_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--estimator', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    kwargs = cmdline.parse_kwargs(parser)
    main(**kwargs)
