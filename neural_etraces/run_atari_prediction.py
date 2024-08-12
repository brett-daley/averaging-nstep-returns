import argparse
from collections import deque
import itertools

import numpy as np

import agents
import cmdline
from tasks import AtariPredictionTask
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
    assert 0 < duration <= 100_000_000

    task = AtariPredictionTask(env)

    agent_cls = getattr(agents, agent)
    agent_args = (task.env.observation_space, task.env.action_space, seed, discount)
    agent = agent_cls(*agent_args, **agent_kwargs)

    # Start training

    performance = np.zeros(duration + 1)
    i = 0
    episodes = 0

    ep_predictions = []
    ep_rewards = []
    recent_rms_errors = deque(maxlen=100)
    score = 0.0

    for t in itertools.count(start=1):
        obs, action, next_obs, reward, terminated = task.step()
        truncated = False
        score += reward  # Add the raw reward here for accurate score

        # Normalize images in [0,1]
        obs = normalize_image(obs)
        next_obs = normalize_image(next_obs)

        reward = np.sign(reward)  # Reward clipping
        ep_rewards.append(reward)

        # Measure prediction before updating parameters to avoid bias
        value = agent.predict(obs).item()
        ep_predictions.append(value)

        agent.reinforce(obs, action, next_obs, reward, terminated, truncated)

        if terminated:
            episodes += 1

            disc_returns = calculate_disc_returns(ep_rewards, discount)
            ep_predictions = np.array(ep_predictions)
            rms_error = np.sqrt(np.mean(np.square(disc_returns - ep_predictions)))

            recent_rms_errors.append(rms_error)
            avg_rms_error = np.mean(recent_rms_errors)

            if verbose:
                print(f"{task.time():.2f}s  t={t}  ep={episodes}  score={score}  error={rms_error} (avg: {avg_rms_error:.2f})")

            while i <= t:
                performance[i] = rms_error
                if i == duration:
                    return performance
                i += 1

            # Reset for next episode
            ep_predictions = []
            ep_rewards = []
            score = 0.0


def normalize_image(image):
    assert image.dtype == np.uint8
    return image.astype(np.float32) / 255.0


def calculate_disc_returns(rewards, discount):
    assert 0.0 <= discount <= 1.0
    disc_returns = []
    G = 0.0
    for r in reversed(rewards):
        G = (discount * G) + r
        disc_returns.append(G)
    disc_returns.reverse()
    return np.array(disc_returns)


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
