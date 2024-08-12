import numpy as np

from returns.numpy.Q import calculate_td_lambda_errors, calculate_n_step_returns, calculate_harmonic_returns


def calculate(estimator: str, v, next_v, rewards, terminateds, truncateds, discount):
    dones = np.logical_or(terminateds, truncateds)  # End of episode, regardless of reason

    bootstraps = np.where(terminateds, 0.0, next_v)
    td_errors = rewards + (discount * bootstraps) - v

    uncut_traces = np.ones_like(rewards)  # For ignoring cuts

    tokens = estimator.split('-')  # Estimator names are strings delimited by '-'

    if tokens[0] == 'lambda':  # Uncorrected lambda-returns
        assert len(tokens) == 2
        lambd = float(tokens[1])
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, uncut_traces)
        return td_lambda_errors + v

    elif tokens[0] == 'nstep':  # n-step returns
        assert len(tokens) == 2
        n = int(tokens[1])
        return calculate_n_step_returns(n, rewards, bootstraps, discount, dones)

    elif tokens[0] == 'harmonic':
        assert len(tokens) == 1
        return calculate_harmonic_returns(rewards, bootstraps, discount, dones)

    elif tokens[0] == 'half':
        assert len(tokens) == 3
        n1 = int(tokens[1])
        n2 = int(tokens[2])
        G1 = calculate_n_step_returns(n1, rewards, bootstraps, discount, dones)
        G2 = calculate_n_step_returns(n2, rewards, bootstraps, discount, dones)
        return (G1 + G2) / 2

    else:
        raise ValueError(f"unknown return estimator '{estimator}'")
