import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import itertools
import os
import sys
import traceback
from typing import Generator, List, Tuple

import numpy as np
import yaml


this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(this_dir, '..')
sys.path.append(os.path.join(root_dir, 'neural_etraces'))


def cartesian_product(d: dict) -> Tuple[dict]:
    for key, values in d.items():
        if not isinstance(values, list):
            d[key] = [values]

    keys = d.keys()
    combos = itertools.product(*d.values())
    return tuple({k: v for k, v in zip(keys, c)} for c in combos)


def import_function(name: str) -> None:
    assert '.' in name
    module_name, function_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def seed_generator(trials: int, meta_seed: int) -> Generator[int, None, None]:
    np_random = np.random.default_rng(meta_seed)
    return (np_random.integers(2**32).item() for _ in range(trials))


class DataFile:
    def __init__(self, function_name: str, kwargs: dict, save_dir: str):
        self.function = import_function(function_name)
        self.kwargs = dict(kwargs)
        self.save_dir = save_dir

        kwargs = dict(self.kwargs)  # Make a copy
        seed = kwargs.pop('seed')

        name = '_'.join([f"{k}-{v}" for k, v in sorted(kwargs.items())])
        name = name.replace('/', '')  # Remove slashes to make path safe
        prefix = str(seed) + '_'
        self.path = os.path.join(save_dir, prefix + name + '.npy')
        self.path_no_prefix = os.path.join(save_dir, name + '.npy')

    def mkdir(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def exists(self) -> bool:
        # Check both paths in case files have already been consolidated
        return os.path.exists(self.path_no_prefix) or os.path.exists(self.path)

    def save(self, array: np.ndarray) -> None:
        np.save(self.path, array)
        print(f"Saved {self.path}")


def load_config(save_dir: str, meta_seed: int) -> List[DataFile]:
    config_path = os.path.join(this_dir, '.automate.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f.read())

    data_files = []
    for experiment in config.values():
        function_name = experiment['function']
        kwargs_list = cartesian_product(experiment['kwargs'])
        trials = experiment['trials']
        for kwargs in kwargs_list:
            for s in seed_generator(trials, meta_seed):
                kwargs['seed'] = s
                data_files.append( DataFile(function_name, kwargs, save_dir) )

    return data_files


def get_partition(data_files, partition_str: str):
    # Parse the partition string
    assert '/' in partition_str
    index, num_partitions = map(int, partition_str.split('/'))

    # Ensure these values are sensible
    assert num_partitions >= 1
    assert 1 <= index <= num_partitions
    assert len(data_files) >= num_partitions

    if num_partitions == 1:
        # If there's only 1 partition, we don't need to split the files
        return data_files

    target_partition = []
    partition_sizes = np.zeros(num_partitions)

    # Sort the data files for reproducibility
    for f in sorted(data_files, key=lambda f: f.path):
        if f.exists():
            continue  # File already exists, skip

        # Add the next largest file to the smallest partition
        i = np.argmin(partition_sizes)
        partition_sizes[i] += 1

        if i + 1 == index:  # +1 to convert to 1-indexing
            # We only build the partition that we want
            target_partition.append(f)

    return target_partition


def print_summary(data_files, overwrite, verbose):
    if verbose:
        for f in data_files:
            print(f"[{'x' if f.exists() else ' '}] {f.path}")
        print()

    total = len(data_files)
    exist = sum([1 for f in data_files if f.exists()])
    todo = total if overwrite else (total - exist)

    digits = len(str(total))
    print(f"{total:>{digits}} Total")

    todo_str = f"{exist:>{digits}} Exist"
    if overwrite:
        todo_str += " (will be overwritten!)"
    print(todo_str)

    print('-' * digits)
    print(f"{todo:>{digits}} To-Do")
    print()
    print("*** This was just a test! No jobs were actually dispatched.")
    print("*** If the output looks correct, re-run with the '--go' argument.")
    print("*** Or, add '--verbose' to see the job list.")
    print(end='', flush=True)


def main(go: bool, max_parallel: int, meta_seed: int, overwrite: bool, save_dir: str, verbose: bool, partition: str) -> None:
    data_files = load_config(save_dir, meta_seed)
    data_files = get_partition(data_files, partition)

    if not go:
        print_summary(data_files, overwrite, verbose)
        return

    # Try to make all necessary directories before dispatching the jobs in case there's an error
    for file in data_files:
        file.mkdir()

    # Disable JAX GPU memory preallocation in case these jobs are sharing a GPU
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    if max_parallel is None:
        max_parallel = len(os.sched_getaffinity(0))  # Number of available CPUs

    future_to_file = {}
    with ProcessPoolExecutor(max_parallel) as executor:
        for file in data_files:
            if file.exists() and not overwrite:
                continue
            future = executor.submit(file.function, **file.kwargs)
            future_to_file[future] = file

        for future in as_completed(future_to_file.keys()):
            file = future_to_file[future]
            try:
                results = future.result()
                file.save(results)
            except Exception as e:
                print("FAILURE generating", file.path)
                print(traceback.format_exc())
            future_to_file.pop(future)  # Pop to save memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--go', action='store_true')
    parser.add_argument('--max_parallel', type=int, default=None)
    parser.add_argument('--meta_seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--partition', type=str, default="1/1")
    kwargs = vars(parser.parse_args())
    main(**kwargs)
