import flax.linen as nn
import jax
import jax.numpy as jnp

from extractors.atari_cnn import AtariCNN
from extractors.extractor import Extractor
from extractors.identity import Identity
from extractors.minatar_cnn import MinAtarCNN
from extractors.mlp import MLP
from utils import normalize_if_image


def make(task_id: str) -> Extractor:
    return {
        'none': Identity(),
        'mlp2l32u': FlaxWrapper(MLP(hidden_units=[32, 32])),
        'mlp2l64u': FlaxWrapper(MLP(hidden_units=[64, 64])),
        'mlp2l512u': FlaxWrapper(MLP(hidden_units=[512, 512])),
        'hardtanh': FlaxWrapper(MLP(hidden_units=[512, 512], activation=nn.hard_tanh)),
        'minatarcnn': FlaxWrapper(MinAtarCNN()),
        'ataricnn': FlaxWrapper(AtariCNN()),
    }[task_id]


class FlaxWrapper(Extractor):
    def __init__(self, model: nn.Module):
        self.model = model
        self._forward = self.model.apply

    def generate_parameters(self, input_shape, prng_key):
        key = prng_key
        key, subkey = jax.random.split(key)

        input_shape = (1, *input_shape)  # Add batch dimension
        params = self.model.init(subkey, jnp.empty(input_shape))

        return params, self.model.outputs, key

    def forward(self, params, x):
        x = normalize_if_image(x)
        return self._forward(params, x)
