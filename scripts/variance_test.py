import sys
sys.path.append('neural_etraces')

import gymnasium as gym
import gym_classics
gym_classics.register('gymnasium')
import matplotlib.pyplot as plt
import numpy as np

from plot import save, set_plot_size
from tasks.dynamic_programming import policy_evaluation


NSTEPS = 21
EPISODES = 10_000


def run(env, discount, seed=0):
    env.action_space.seed(seed)

    n = env.action_space.n
    behavior_policy = lambda s: np.ones(n) / n
    V = policy_evaluation(env, discount, behavior_policy, precision=1e-9)
    V.setflags(write=False)  # Read-only
    print("v_pi =", V)

    seeded_env = False

    batch_td_errors = []
    for _ in range(EPISODES):  # For each episode
        td_errors = []
        if not seeded_env:
            state, _ = env.reset(seed=seed)
            seeded_env = True
        else:
            state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            assert not truncated

            delta = reward - V[state]
            if not done:
                delta += discount * V[next_state]
            td_errors.append(delta)

            state = next_state

        # print(td_errors)
        batch_td_errors.append(td_errors)

    shortest = min([len(episode) for episode in batch_td_errors])
    longest = max([len(episode) for episode in batch_td_errors])
    # Zero pad each episode
    batch_td_errors = [episode + [0] * (longest - len(episode)) for episode in batch_td_errors]
    return np.array(batch_td_errors), shortest


def plot(env_id, discount, y_max, title=None, use_ylabel=True):
    env = gym.make(env_id)

    td_errors, shortest = run(env, discount)
    print("shortest path length =", shortest)

    variances = []
    for n in range(1, NSTEPS + 1):
        first_n_errors = td_errors[:, :n]
        assert first_n_errors.shape[1] == n
        weights = np.array([pow(discount, i) for i in range(n)])[None]
        nstep_errors = np.sum(weights * first_n_errors, axis=1)
        variances.append(
            np.var(nstep_errors, ddof=1)
        )
    variances = np.array(variances)

    plt.figure()

    x = 1 + np.arange(NSTEPS, dtype=np.int32)
    green = '#27ae60'
    plt.plot(x, variances, color=green, label='observed')

    opt_line = variances[0] * x
    pess_line = variances[0] * np.square(x)
    plt.plot(x, opt_line, linestyle='--', linewidth=0.75, color='black', label="optimistic")
    plt.plot(x, pess_line, linestyle='--', linewidth=0.75, color='black', label="pessimistic")

    plt.xlim([1, NSTEPS])
    plt.xticks([1, 6, 11, 16, 21])
    plt.ylim([0, y_max])

    if title is not None:
        plt.title(title)
    plt.xlabel("$n$-step")

    if use_ylabel:
        plt.ylabel("Variance")
    else:
        plt.ylabel(" ")

    set_plot_size()

    save(f"variance_{env_id}", 'plots', pdf=True)


if __name__ == '__main__':
    plt.style.use('custom.mplstyle')

    plot("19Walk-v0", discount=1.0, y_max=0.25, title="19-State Random Walk")
    plot("ClassicGridworld-v0", discount=1.0, y_max=0.5, title=r"$4 \times 3$ Gridworld", use_ylabel=False)
    plot("SparseGridworld-v0", discount=0.99, y_max=0.1, title=r"$10 \times 8$ Gridworld", use_ylabel=False)
