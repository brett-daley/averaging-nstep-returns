from argparse import ArgumentParser
from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from plot import format_plot, save
from summarize import calculate_auc
from lambda_sweep import reduce_max_auc


def com_sweep(input_dir, output_dir, coms, estimator_lists, labels, pattern, name="com-sweep"):
    plt.figure()
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'font.size': 22})

    pattern = os.path.join(input_dir, pattern)

    for est_list, label in zip(estimator_lists, labels):
        paths, means, conf95s = zip(*[reduce_max_auc(pattern.format(est), reduce='mean') for est in est_list])

        print("Generating line from these files:")
        for x in paths:
            print(' ', x)

        xs = coms
        ys, conf95s = map(np.array, [means, conf95s])

        plt.plot(xs, ys, linewidth=1.5, label=label)
        plt.fill_between(xs, (ys - conf95s), (ys + conf95s), alpha=0.25, linewidth=0)

    plt.xlabel("Theoretical COM")
    plt.ylabel("Normalized AUC")
    plt.xlim([0, max(coms)])
    format_plot(aspect=2, legend=True)
    save(name, output_dir, pdf=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    coms = [2, 3, 5, 10, 20, 25]
    nstep_estimators = ['ql-nstep-2', 'ql-nstep-3', 'ql-nstep-5', 'ql-nstep-10', 'ql-nstep-20', 'ql-nstep-25']
    lambda_estimators = ['ql-peng-0.5', 'ql-peng-0.67', 'ql-peng-0.8', 'ql-peng-0.9', 'ql-peng-0.95', 'ql-peng-0.96']
    labels = [r"$n$-step return", r"$\lambda$-return"]

    com_sweep(
        args.input_dir,
        args.output_dir,
        coms,
        [nstep_estimators, lambda_estimators],
        labels,
        pattern="agent-FVR_cpu-True_defaults-lunar_estimator-{}_lr-*.npy",
        name="com-sweep_lunar"
    )
