from schedules.schedules import *


def make(schedule_name: str) -> Schedule:
    try:
        epsilon = float(schedule_name)
        return lambda t: epsilon
    except ValueError:
        pass

    return {
        'toy-control': ExponentialSchedule(1.0, 0.01, decay_factor=pow(0.999, 1 / 100)),
        'minatar': LinearSchedule(1.0, 0.1, timeframe=100_000),
        'atari': LinearSchedule(1.0, 0.1, timeframe=1_000_000),
    }[schedule_name]
