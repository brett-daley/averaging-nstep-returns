from argparse import ArgumentParser
from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np

from plot import set_plot_size, save
from summarize import calculate_auc


def alpha_sweep(input_dir, output_dir, alpha_values, patterns, labels, colors, exclude_first_n=0, title=None, use_ylabel=False, ylim=None, name=None):
    plt.figure()

    for p, label, color in zip(patterns, labels, colors):
        p = os.path.join(input_dir, p)

        means, conf95s, _ = zip(*[
            calculate_auc(p.format(a), reduce='mean', exclude_first_n=exclude_first_n)
            for a in alpha_values
        ])

        xs = alpha_values
        ys, conf95s = map(np.array, [means, conf95s])

        plt.plot(xs, ys, label=label, color=color)
        plt.fill_between(xs, (ys - conf95s), (ys + conf95s), alpha=0.25, linewidth=0, color=color)
        max_line = np.max(ys) * np.ones_like(xs)
        plt.plot(xs, max_line, linestyle='--', linewidth=0.75, color=color)

    plt.xlabel(r"$\alpha$")
    if use_ylabel:
        plt.ylabel("Normalized AUC")
    else:
        plt.ylabel(" ")

    plt.xlim(min(alpha_values), max(alpha_values))
    if title is not None:
        plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xscale('log')
    set_plot_size(aspect=1)
    plt.legend(loc='best')
    save(name, output_dir, pdf=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    plt.style.use('custom.mplstyle')

    plot_func = partial(alpha_sweep, args.input_dir, args.output_dir)

    # MinAtar
    alpha_values = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    for env, ylim in zip(("Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"),
                         ([0, 20], [0, 20], [0, 60], [0, 50], [0, 80])):
            for n in [3, 5]:
                for use_ylabel in [False, True]:
                    name = f"alpha-sweep_{env}_n{n}"
                    if use_ylabel:
                        name += "_ylabel"
                    plot_func(
                        alpha_values,
                        patterns=[
                            f"agent-ALR2_batch_len-1_cpu-True_defaults-minatar_env-MinAtar{env.replace(' ', '')}-v1_est-nstep-{n}_lr-{{}}.npy",
                            f"agent-ALR2_batch_len-1_cpu-True_defaults-minatar_env-MinAtar{env.replace(' ', '')}-v1_est-pilar2-{n}_lr-{{}}.npy",
                        ],
                        labels=[
                            f"$n={n}$",
                            f"PiLaR(${n}$)"
                        ],
                        colors=['#2980b9', '#c0392b'],
                        exclude_first_n=4_000_000,  # Ignore initial exploration phase
                        title=env,
                        use_ylabel=use_ylabel,
                        ylim=ylim,
                        name=name
                    )
                plt.close()
