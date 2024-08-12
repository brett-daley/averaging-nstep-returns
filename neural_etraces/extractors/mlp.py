from collections.abc import Iterable
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    hidden_units: Sequence[int]
    activation: Callable[[jnp.array], jnp.array] = nn.relu

    def setup(self):
        assert isinstance(self.hidden_units, Iterable), "must be list or tuple"
        assert self.hidden_units, "must have at least one hidden layer"

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_units:
            x = nn.Dense(h)(x)
            x = self.activation(x)
        return x

    @property
    def outputs(self):
        return self.hidden_units[-1]
