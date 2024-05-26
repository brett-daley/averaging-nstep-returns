import numpy as np
import jax
import jax.numpy as jnp


class RandomAccessBuffer:
    def __init__(self, maxlen):
        self._buffer = []
        self._maxlen = maxlen
        self._write_ptr = 0

    def __getitem__(self, i):
        return self._buffer[i]

    def __len__(self):
        return len(self._buffer)

    def append(self, obj):
        if len(self._buffer) < self._maxlen:
            self._buffer.append(obj)
        else:
            self._buffer[self._write_ptr] = obj
            self._write_ptr = (self._write_ptr + 1) % self._maxlen

    def sequence(self, start, length):
        assert 0 <= start + length <= len(self)
        for i in range(length):
            yield self._buffer[(self._write_ptr + start + i) % len(self)]

    def all(self):
        return tuple(self.sequence(start=0, length=len(self)))

    def clear(self):
        self._buffer.clear()
        self._write_ptr = 0


class ReplayMemory:
    def __init__(self, capacity: int, seed: int):
        self._np_random = np.random.default_rng(seed)
        self._deque = RandomAccessBuffer(maxlen=capacity)

    def __len__(self):
        return len(self._deque)

    def append(self, x):
        self._deque.append(x)

    def sample_minibatch(self, batch_size):
        indices = self._np_random.integers(len(self._deque), size=batch_size)
        minibatch = tuple(self._deque[i] for i in indices)
        return self.as_arrays(minibatch)

    def sample_trajectories(self, batch_size, length):
        batch = []
        for _ in range(batch_size):
            start = self._np_random.integers(len(self) - length + 1)
            trajectory = tuple(self._deque.sequence(start, length))
            batch.append(self.as_arrays(trajectory))
        return tuple(map(jnp.stack, zip(*batch)))

    # def build_cache(self, cache_size, block_size):
    #     assert len(self) >= block_size
    #     assert (cache_size % block_size) == 0
    #     num_blocks = cache_size // block_size
    #     cache = []
    #     for _ in range(num_blocks):
    #         start = self._np_random.integers(len(self) - block_size + 1)
    #         block = list(self._deque.sequence(start, length=block_size))
    #         obs, action, next_obs, reward, term, _ = block[-1]
    #         block[-1] = (obs, action, next_obs, reward, term, True)
    #         cache.extend(block)
    #     return self.as_arrays(cache)

    def get_all(self):
        batch = self._deque.all()
        return self.as_arrays(batch)

    def clear(self):
        self._deque.clear()

    @staticmethod
    def as_arrays(batch):
        return tuple(map(jnp.array, zip(*batch)))
