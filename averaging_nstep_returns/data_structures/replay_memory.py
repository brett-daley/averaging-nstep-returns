from abc import ABC
from operator import itemgetter

import numpy as np


class ReplayMemory(ABC):
    def __init__(self, capacity, seed):
        self.memory = CoreMemory(capacity)
        self.save = self.memory.save
        self.np_random = np.random.default_rng(seed)

    def __len__(self):
        return len(self.memory)

    def sample_trajectories(self, batch_size=1, length=1):
        assert batch_size >= 1
        assert length >= 1
        indices = np.stack([
            np.arange(base, base + length, dtype=np.int32)
            # -1 because (end_index - start_index) = length - 1
            for base in self._random_minibatch_indices(batch_size, except_last=(length - 1))])
        return self.memory.get(indices)

    def _random_index(self, except_last=0):
        assert 0 <= except_last < len(self)
        return self.np_random.integers(len(self) - except_last)

    def _random_minibatch_indices(self, batch_size, except_last=0):
        return (self._random_index(except_last) for _ in range(batch_size))


class CoreMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffers = []
        self.allocated = False
        self.write_ptr = 0
        self.num_writes = 0

    def __len__(self):
        return min(self.num_writes, self.capacity)

    def _allocate(self, array):
        array = np.array(array)
        return np.empty([self.capacity, *array.shape], dtype=array.dtype)

    def save(self, *args):
        if not self.allocated:
            for array in args:
                self.buffers.append(self._allocate(array))
            self.allocated = True

        p = self.write_ptr
        for i, array in enumerate(args):
            self.buffers[i][p] = array

        self.write_ptr = (p + 1) % self.capacity
        self.num_writes += 1

    def get(self, indices: np.ndarray):
        assert len(self) > 0
        assert (0 <= indices).all()
        assert (indices < len(self)).all()
        indices = (self.write_ptr + indices) % len(self)  # Shift by pointer
        return tuple(map(itemgetter(indices), self.buffers))
