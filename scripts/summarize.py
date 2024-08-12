from argparse import ArgumentParser
from operator import itemgetter
import os

import numpy as np
from tabulate import tabulate


def calculate_auc(path, reduce='sum', exclude_first_n=0):
    assert reduce in {'sum', 'mean'}
    performance = np.load(path)
    assert performance.ndim == 2

    performance = performance[:, exclude_first_n:]
    if reduce == 'sum':
        AUCs = np.sum(performance, axis=1)
    else:
        AUCs = np.mean(performance, axis=1)
    mean_auc = np.mean(AUCs)

    conf95_auc = 0.0
    n = len(AUCs)
    if n > 1:
        std_auc = np.std(AUCs, ddof=1)
        conf95_auc = 1.96 * std_auc / np.sqrt(n)

    return mean_auc, conf95_auc, n


def main(directory: str, contains: str, reverse: bool, reduce: str, exclude: int = 0):
    table = []
    for file in os.listdir(directory):
        if not contains in file:
            continue
        path = os.path.join(directory, file)
        mean_auc, conf95_auc, n = calculate_auc(path, reduce, exclude_first_n=exclude)
        table.append([file, n, mean_auc, conf95_auc])

    table = sorted(table, key=lambda x: float(itemgetter(2)(x)), reverse=reverse)
    print(tabulate(table, headers=["Experiment", "Trials", "AUC", "+/- 95% Conf."]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('--contains', type=str, default='')
    parser.add_argument('--reverse', action='store_false')
    parser.add_argument('--reduce', default='sum')
    parser.add_argument('--exclude', type=int, default=0)
    kwargs = vars(parser.parse_args())
    main(**kwargs)
