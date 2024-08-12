import flax.linen as nn


class MinAtarCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        if x.ndim == 3:
            x = x[None]
        assert x.ndim == 4, "input must have shape (N, H, W, C) or (H, W, C)"

        x = nn.Conv(16, kernel_size=[3, 3], strides=1, padding='VALID')(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(self.outputs)(x)
        x = nn.relu(x)
        return x

    @property
    def outputs(self):
        return 128
