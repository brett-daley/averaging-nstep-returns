from argparse import ArgumentParser
from glob import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml


def set_plot_size(aspect=1):
    ax = plt.gca()
    ax.set_aspect(1.0 / (aspect * ax.get_data_ratio()))

    fig = plt.gcf()
    y = 4.8
    x = aspect * y
    fig.set_size_inches(x, y)

    if aspect > 1:
        fig.tight_layout(pad=0)
    else:
        fig.tight_layout(pad=0.1)


def get_metric_suffix_formatter():
    def format_func(x, pos):
        if 1_000 <= abs(x) < 1_000_000:
            x = str(int(x / 1_000))
            x += 'k'
        elif 1_000_000 <= abs(x):
            x = str(int(x / 1_000_000))
            x += 'M'
        else:
            x = int(x)
        return x

    return matplotlib.ticker.FuncFormatter(format_func)


def save(name, directory, pdf):
    if not os.path.exists(directory):
        os.mkdir(directory)

    path = os.path.join(directory, name)
    if pdf:
        path += '.pdf'
        plt.savefig(path, format='pdf')
    else:
        path += '.png'
        plt.savefig(path, format='png')
    print(f"Saved plot as {path}", flush=True)


def set_plot_attributes(params):
    if 'title' in params:
        plt.title(params['title'])

    if 'xlabel' in params:
        plt.xlabel(params['xlabel'])

    if 'ylabel' in params:
        plt.ylabel(params['ylabel'])

    if 'xlim' in params:
        plt.xlim(params['xlim'])

    if 'ylim' in params:
        plt.ylim(params['ylim'])

    if ('ylog' in params) and params['ylog']:
        plt.yscale('log')


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="scripts/.plot.yaml")
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='plots')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    plt.style.use('custom.mplstyle')

    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh.read())

    for plot_group, group_params in config.items():
        directory = group_params['dir']

        for plot_name, plot_params in group_params['plots'].items():
            plot_path = f"plots/{plot_name}.png"

            if args.only is not None and args.only != plot_path:
                continue

            if os.path.exists(plot_path) and not args.overwrite:
                print(f"{plot_path} already exists (add --overwrite to modify)")
                continue

            plt.figure()
            set_plot_attributes(group_params)
            downsample = group_params.get('downsample', 1)

            for i, (path, color) in enumerate(zip(plot_params['experiments'], group_params['colors'])):
                set_plot_attributes(plot_params)  # Overrides group parameters

                if 'labels' in plot_params:
                    label = plot_params['labels'][i]
                else:
                    label = group_params['labels'][i]

                path = os.path.join(directory, path)
                if '*' in path:  # Wildcard
                    matches = glob(path)
                    assert len(matches) > 0, f"{path} did not match anything"
                    assert len(matches) == 1, f"{path} must be unambiguous"
                    path = matches[0]

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
                    plt.fill_between(x, (y - conf95), (y + conf95), color=color, alpha=0.25, linewidth=0)

                plt.plot(x, y, color=color, label=label)

            formatter = get_metric_suffix_formatter()
            plt.gca().xaxis.set_major_formatter(formatter)

            if 'legend' in plot_params:
                plt.legend(loc=plot_params['legend'])
            elif 'legend' in group_params:
                plt.legend(loc=group_params['legend'])

            set_plot_size()
            save(plot_name, args.output_dir, pdf=args.pdf)

            plt.ylabel(" ")
            save(plot_name + '_no_ylabel', args.output_dir, pdf=args.pdf)

            plt.close()


if __name__ == '__main__':
    main()
