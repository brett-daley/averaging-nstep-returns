from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np

from plot import get_metric_suffix_formatter, save, set_plot_size


INPUT_DIR = 'results'
OUTPUT_DIR = 'plots'


def main():
    parser = ArgumentParser()
    parser.add_argument('files', type=str, nargs='+')
    args = parser.parse_args()

    plt.style.use('custom.mplstyle')

    plt.figure()

    for f in args.files:
        path = os.path.join(INPUT_DIR, f)
        downsample = 1

        performance = np.load(path)
        assert performance.ndim == 2
        n = performance.shape[0]
        T = performance.shape[1]
        performance = performance[:, ::downsample]

        x = np.arange(T)[::downsample]
        y = np.mean(performance, axis=0)

        if n > 1:
            std = np.std(performance, ddof=1, axis=0)
            conf95 = 1.96 * std / np.sqrt(n)
            plt.fill_between(x, (y - conf95), (y + conf95), alpha=0.25, linewidth=0)

        plt.plot(x, y)

    formatter = get_metric_suffix_formatter()
    plt.gca().xaxis.set_major_formatter(formatter)

    set_plot_size()
    save('quickplot', OUTPUT_DIR, pdf=False)
    plt.close()


if __name__ == '__main__':
    main()
