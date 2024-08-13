import jax.numpy as jnp

from averaging_nstep_returns.extractors import Extractor


class Identity(Extractor):
    def generate_parameters(self, input_shape, prng_key):
        params = ()
        features = input_shape[0]
        return params, features, prng_key

    def forward(self, params, x):
        return x
