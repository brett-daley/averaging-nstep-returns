from contextlib import nullcontext
import os

import jax
import jax.numpy as jnp
import numpy as np


def egreedy(x, epsilon):
    """epsilon-greedy distribution along axis=-1"""
    assert x.ndim == 2
    assert 0.0 <= epsilon <= 1.0
    n = x.shape[-1]

    argmax_actions = jnp.argmax(x, axis=-1)
    greedy = jax.nn.one_hot(argmax_actions, num_classes=n)

    uniform = jnp.ones_like(x) / n

    return (1-epsilon) * greedy + epsilon * uniform


def jax_device_context(force_cpu=False):
    if not force_cpu:
        return nullcontext()
    return jax.default_device(jax.devices('cpu')[0])


def use_deterministic_gpu_ops():
    # See https://github.com/google/jax/discussions/10674
    os.environ['XLA_FLAGS'] = "--xla_gpu_deterministic_ops=true"
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def onehot(index, size, dtype):
    x = np.zeros(size, dtype)
    x[index] = 1.0
    return x


def normalize_if_image(obs):
    if obs.dtype != jnp.uint8:
        return obs
    return obs.astype(jnp.float32) / 255.0


def print_vf(v, task):
    from gym_classics.envs.abstract.gridworld import Gridworld

    env = task.env
    if isinstance(env, Gridworld):
        v_grid = np.zeros(task.dims)
        for s in env.states():
            x, y = env._decode(s)
            v_grid[x,y] = v[s]
        v_grid = v_grid.T
        print(v_grid)

    else:
        print(v)
