from argparse import ArgumentParser
from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from plot import format_plot, save
from summarize import calculate_auc


def reduce_max_auc(pattern, reduce='sum'):
    files = glob(pattern)
    assert files

    best = None, -float('inf'), None
    for f in files:
        mean_auc, conf95_auc, _ = calculate_auc(f, reduce)
        if mean_auc > best[1]:
            best = f, mean_auc, conf95_auc

    return best


def lambda_sweep(input_dir, output_dir, lambda_values, patterns, name="lambda-sweep"):
    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    for p in patterns:
        p = os.path.join(input_dir, p)

        paths, means, conf95s = zip(*[reduce_max_auc(p.format(lambd=lambd)) for lambd in lambda_values])

        print("Generating plot from these files:")
        for x in paths:
            print(x)

        xs = lambda_values
        ys, conf95s = map(np.array, [means, conf95s])

        plt.plot(xs, ys, linewidth=1.5)
        plt.fill_between(xs, (ys - conf95s), (ys + conf95s), alpha=0.25, linewidth=0)

    plt.xlabel("$\lambda$")
    plt.ylabel("Area Under the Curve")
    plt.xlim([0, 1])
    format_plot(aspect=2, legend=False)
    save(name, output_dir, pdf=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    lambda_sweep(
        args.input_dir,
        args.output_dir,
        lambda_values=[0, 0.5, 0.7, 0.8, 0.9],
        patterns=[
            "agent-FVR_defaults-toy-control_env-CartPole-v1_estimator-watkins-{lambd}_lr-*.npy",
        ],
        name="lambda-sweep_cartpole"
    )
