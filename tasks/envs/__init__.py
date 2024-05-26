import gymnasium as gym
from gymnasium.envs import register
from gymnasium.wrappers import TimeLimit
# import gym_classics
# gym_classics.register('gymnasium')
# from gym_classics.envs.abstract.gridworld import Gridworld

from tasks.envs import wrappers


def make(env_id: str):
    if env_id.startswith('ALE'):
        return make_atari_env(env_id)

    env = gym.make(env_id)

    override_limits = {
        'Acrobot-v1': 500,
        'MountainCar-v0': 1_000,
    }
    if env_id in override_limits:
        assert isinstance(env, TimeLimit)
        env = env.env  # Remove existing TimeLimit wrapper
        env = TimeLimit(env, max_episode_steps=override_limits[env_id])

    env = wrappers.EnvProbe(env)
    return env


def make_atari_env(env_id: str):
    env = gym.make(env_id, obs_type='rgb', frameskip=1, repeat_action_probability=0.0, full_action_space=False)
    # 108k frames @ 60 Hz = 30 minutes of gameplay
    env = TimeLimit(env, max_episode_steps=108_000)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = wrappers.FireReset(env)
    env = wrappers.RandomNoOpReset(env)

    # To avoid miscounts, probe must come after time limit and no-ops
    # but before reward clipping and episodic-life resets
    env = wrappers.EnvProbe(env)

    env = wrappers.LifeLossTermination(env)
    env = wrappers.AtariPreprocessing(env)
    env = wrappers.History(env, history_len=4)

    # Reward clipping must come after frame skipping
    env = wrappers.AtariRewardClipping(env)
    return env
