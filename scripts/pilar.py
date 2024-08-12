import sys
sys.path.append('neural_etraces')

import matplotlib.pyplot as plt
import numpy as np

from agents.control.alr import get_pilar_weight_func, best_approximation
from plot import save, set_plot_size


def main(n, discount):
    (n1, n2, w), _ = best_approximation(n, discount)
    print(f"n1={n1}, n2={n2}, w={w}")
    lambd = (1 - pow(discount, n-1)) / (1 - pow(discount, n))

    N = 50
    x_axis = np.arange(N + 1)
    pilar_weight_func, _ = get_pilar_weight_func(n, discount, n1, n2)
    pilar_weights = np.array([pilar_weight_func(i) for i in range(N + 1)])
    lambda_weights = np.power(discount * lambd, x_axis)

    plt.xlim(0, N)

    margin = 0.01
    plt.ylim(-margin, 1 + margin)

    plt.plot(x_axis, lambda_weights, linestyle='--', linewidth=0.5, marker='.', color='#7f8c8d', label="$\lambda$-return")
    plt.plot(x_axis, pilar_weights, linestyle='--', linewidth=0.5, marker='.', color='#c0392b', label="PiLaR")

    plt.xlabel("Time since state visitation")
    plt.ylabel("TD-error weight")

    plt.legend(loc="upper right")

    set_plot_size()
    name = f"pilar_{n}"
    directory = 'plots'
    save(name, directory, pdf=False)
    save(name, directory, pdf=True)


if __name__ == '__main__':
    plt.style.use('custom.mplstyle')
    main(n=10, discount=0.99)
