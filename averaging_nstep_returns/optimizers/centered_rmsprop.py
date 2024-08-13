from jax.example_libraries.optimizers import Schedule, make_schedule, optimizer
import jax.numpy as jnp


@optimizer
def centered_rmsprop(lr: Schedule, alpha: float = 0.95, epsilon: float = 0.01):
    lr = make_schedule(lr)
    assert 0.0 <= alpha < 1.0
    assert 0.0 < epsilon

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state

        m = alpha * m + (1-alpha) * g
        v = alpha * v + (1-alpha) * jnp.square(g)
        variance = jnp.sqrt(v - jnp.square(m))

        step = lr(i) * g / (variance + epsilon)
        x -= step

        return x, m, v

    def get_params(state):
        return state[0]

    return init, update, get_params
