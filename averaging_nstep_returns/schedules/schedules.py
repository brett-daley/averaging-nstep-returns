from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def __call__(self, t):
        raise NotImplementedError


class LinearSchedule(Schedule):
    def __init__(self, init_value, final_value, timeframe):
        assert init_value >= final_value
        assert timeframe > 0
        self.init_value = init_value
        self.final_value = final_value
        self.timeframe = timeframe
        self.slope = (final_value - init_value) / timeframe

    def __call__(self, t):
        assert t >= 0
        return max(self.init_value + self.slope * t, self.final_value)


class ExponentialSchedule:
    def __init__(self, init_value, final_value, decay_factor):
        assert init_value >= final_value
        assert 0.0 < decay_factor < 1.0
        self.init_value = init_value
        self.final_value = final_value
        self.decay_factor = decay_factor

    def __call__(self, t):
        assert t >= 0
        return max(self.init_value * pow(self.decay_factor, t), self.final_value)
