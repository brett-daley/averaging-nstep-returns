from abc import ABC, abstractmethod


class Extractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_parameters(self, input_shape, prng_key):
        raise NotImplementedError

    @abstractmethod
    def forward(self, params, x):
        raise NotImplementedError
