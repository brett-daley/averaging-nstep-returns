import argparse

import numpy as np

import agents
import cmdline
from tasks import PredictionTask
from utils import jax_device_context, print_vf, use_deterministic_gpu_ops


def main(**kwargs):  # Hook for automation
    kwargs = cmdline.insert_defaults(kwargs)
    cmdline.assert_not_none(kwargs)

    force_cpu = kwargs.pop('cpu', False)
    if not force_cpu:
        use_deterministic_gpu_ops()

    with jax_device_context(force_cpu):
        return run(**kwargs)


def run(env: str, agent: str, discount: float, duration: float, seed: int, verbose: bool = False, **agent_kwargs):
    duration = int(duration)
    assert duration > 0

    task = PredictionTask(env, discount, duration, seed)

    if verbose:
        print("v_pi =")
        print_vf(task.v_pi, task)

    agent_cls = getattr(agents, agent)
    agent_args = (task.env.observation_space, task.env.action_space, seed, task.discount)
    agent = agent_cls(*agent_args, **agent_kwargs)

    # Start training

    obs, _ = task.reset()

    performance = np.zeros(duration + 1)
    performance[0] = task.msve(agent)
    if verbose:
        print(0, performance[0])

    for t in range(1, duration + 1):
        action, b_prob, t_prob = task.policy()
        next_obs, reward, terminated, truncated, _ = task.step(action)
        agent.reinforce(obs, action, next_obs, reward, terminated, truncated)

        if terminated or truncated:
            next_obs, _ = task.reset()

        obs = next_obs

        performance[t] = task.msve(agent)
        if verbose:
            if (t % 500) == 0:
                print(t, performance[t])

    return performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--env', type=str)
    parser.add_argument('--agent', type=str)
    parser.add_argument('--defaults', type=str)
    parser.add_argument('--discount', type=float)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    kwargs = cmdline.parse_kwargs(parser)
    main(**kwargs)
