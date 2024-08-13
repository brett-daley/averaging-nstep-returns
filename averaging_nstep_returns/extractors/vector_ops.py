import jax.numpy as jnp
from jax.tree_util import tree_map


def add(v1, v2):
    return tree_map(lambda a, b: a + b, v1, v2)


def sub(v1, v2):
    return tree_map(lambda a, b: a - b, v1, v2)


def scale(factor: float, v):
    return tree_map(lambda a: factor * a, v)


def zeros_like(v):
    return tree_map(jnp.zeros_like, v)


def conditional_zeros_like(v, condition: bool):
    return tree_map(lambda a: jnp.where(condition, zeros_like(a), a), v)


def copy(v):
    return tree_map(jnp.copy, v)


def etrace(z, decay: float, increment: dict, rho: float = 1.0):
    """Convenience function for eligibility traces."""
    z = scale(decay, z)
    z = add(z, increment)
    return scale(rho, z)
