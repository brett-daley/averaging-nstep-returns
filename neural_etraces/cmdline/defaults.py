import os
import warnings

import yaml


def assert_not_none(kwargs: dict):
    for key, value in kwargs.items():
        assert value is not None, f"missing value, got {key}={value}"


def insert_defaults(kwargs: dict):
    config_name = kwargs.pop('defaults', None)

    if config_name is not None:
        default_kwargs = load_defaults(config_name)

        for key, default_value in default_kwargs.items():
            should_add = False

            if key not in kwargs:
                should_add = True
            else:
                supplied = kwargs[key]
                if supplied is None:
                    should_add = True
                elif supplied != default_value:
                    warnings.warn(f"overwriting default '{key}' ({default_value}) with new value ({supplied})")

            if should_add:
                kwargs[key] = default_value

    return kwargs


def load_defaults(config_name: str):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(this_dir, 'defaults.yaml')
    with open(path, 'r') as f:
        configs = yaml.safe_load(f.read())
        return configs[config_name]
