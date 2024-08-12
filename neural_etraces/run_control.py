import argparse
import itertools

import numpy as np

import agents
import cmdline
from tasks import envs
from utils import jax_device_context, use_deterministic_gpu_ops


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
    assert 0.0 <= discount <= 1.0

    # Make environment
    env = envs.make(env)
    env.action_space.seed(seed)

    # Make agent
    agent_cls = getattr(agents, agent)
    agent_args = (env.observation_space, env.action_space, seed, discount)
    agent = agent_cls(*agent_args, **agent_kwargs)

    # Start training

    time_periods = 0
    period_start = env.time()
    period_length_minutes = 15

    obs, _ = env.reset(seed=seed)  # Pass seed to initialized RNG

    performance = np.zeros(duration + 1)
    i = 0

    for t in itertools.count(start=1):
        action, b_prob = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.reinforce(obs, action, next_obs, reward, terminated, truncated, b_prob)

        if env.is_done():
            avg_undisc_return = np.mean(env.get_episode_returns()[-100:])

            while i <= t:
                performance[i] = avg_undisc_return

                if (i % 500) == 0 and verbose:
                    episodes = env.get_total_episodes()
                    last_episode_return = env.get_episode_returns()[-1]
                    print(f"{env.time():.2f}s  t={i}  ep={episodes}  {last_episode_return} (avg: {avg_undisc_return:.2f})")

                if i == duration:
                    return performance
                i += 1

            next_obs, _ = env.reset()

        obs = next_obs

        # Periodic logging even when verbose=False (for time estimation on Compute Canada)
        minutes = (env.time() - period_start) / 60
        if minutes >= period_length_minutes:
            time_periods += 1
            percent_complete = round(100 * t / duration, 1)
            print(f"Approximately {round(time_periods * period_length_minutes, 1)} minutes elapsed; "
                  f"{t}/{duration} timesteps ({percent_complete}%) completed")
            period_start = env.time()


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
