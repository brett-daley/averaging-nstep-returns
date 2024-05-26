import numpy as np


def select_axis1(M, indices):
    assert M.ndim == 2
    assert indices.ndim == 1
    n = len(indices)
    assert len(M) == n
    return M[np.arange(n), indices]


def calculate(estimator: str, Q, actions, next_Q, rewards, terminateds, truncateds, b_probs, discount, epsilon):
    dones = np.logical_or(terminateds, truncateds)  # End of episode, regardless of reason

    q_taken = select_axis1(Q, actions)
    v = np.max(Q, axis=-1)
    not_greedy = (q_taken != v)  # Times when an exploratory action was taken

    policy_probs = egreedy_probabilities(next_Q, epsilon)
    next_v = np.sum(policy_probs * next_Q, axis=-1)

    t_probs = select_axis1(policy_probs, actions)
    rhos = t_probs / b_probs

    bootstraps = np.where(terminateds, 0.0, next_v)
    td_errors = rewards + (discount * bootstraps) - q_taken

    cut_traces = np.where(not_greedy, 0.0, 1.0)  # For cutting when exploratory non-greedy action is taken
    uncut_traces = np.ones_like(cut_traces)  # For ignoring cuts

    tokens = estimator.split('-')  # Estimator names are strings delimited by '-'

    if tokens[1] == 'naive':  # Uncorrected lambda-returns
        lambd = float(tokens[2])
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, uncut_traces)
        return td_lambda_errors + q_taken

    elif tokens[1] == 'watkins':  # Watkins' lambda-returns
        assert tokens[0] == 'ql'
        lambd = float(tokens[2])
        # NOTE: This is not true IS here, since we're clipping to 1 instead of using 1/mu.
        # Watkins' Q(lambda) is essentially TB(lambda) when the target policy is greedy.
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, cut_traces)
        return td_lambda_errors + q_taken

    elif tokens[1] == 'peng':  # Peng's lambda-returns
        assert tokens[0] == 'ql'
        lambd = float(tokens[2])
        td_errors += q_taken - v
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, uncut_traces)
        return td_lambda_errors + v

    elif tokens[1] == 'new':  # New lambda-return, equivalent to Peng's with added trace cuts
        assert tokens[0] == 'ql'
        lambd = float(tokens[2])
        td_errors += q_taken - v
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, cut_traces)
        return td_lambda_errors + v

    elif tokens[1] == 'nstep':  # n-step returns
        n = int(tokens[2])
        return calculate_n_step_returns(n, rewards, bootstraps, discount, dones)

    elif tokens[1] == 'hybrid':  # Hybrid returns: n-step delay added to Peng's
        lambd = float(tokens[2])
        n = int(tokens[3])
        td_errors += q_taken - v
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, uncut_traces)
        lambda_returns = td_lambda_errors + v
        return calculate_hybrid_returns(n, lambda_returns, rewards, bootstraps, discount, dones)

    elif tokens[1] == 'is':  # Importance Sampling
        lambd = float(tokens[2])
        td_lambda_errors = calculate_td_lambda_errors(lambd, td_errors, discount, dones, rhos)
        return td_lambda_errors + q_taken

    elif tokens[1] == 'rbis':  # Recency-Bounded Importance Sampling
        lambd = float(tokens[2])
        rbis_errors = calculate_rbis_errors(lambd, td_errors, discount, dones, rhos)
        return rbis_errors + q_taken

    elif tokens[1] == 'harmonic':
        return calculate_harmonic_returns(rewards, bootstraps, discount, dones)

    else:
        raise ValueError(f"unknown return estimator '{estimator}'")


def calculate_n_step_returns(n, rewards, bootstraps, discount, should_cut):
    assert n >= 1
    T = len(rewards)

    one_step_returns = rewards + (discount * bootstraps)
    if n == 1:
        return one_step_returns

    n_step_returns = np.copy(one_step_returns)
    for t in range(T):
        start = t
        end = min(t + n - 1, T - 1)

        G = one_step_returns[end]
        for i in reversed(range(start, end)):
            G = np.where(should_cut[i], bootstraps[i], G)  # Cut for off-policy corrections
            G = rewards[i] + (discount * G)

        n_step_returns[t] = G

    return n_step_returns


def calculate_td_lambda_errors(lambd, td_errors, discount, dones, rhos):
    assert 0.0 <= lambd <= 1.0
    T = len(td_errors)

    td_lambda_errors = np.copy(td_errors)
    weights = np.where(dones[:-1], 0.0, discount * lambd * rhos[1:])  # Cut for off-policy corrections

    E = td_lambda_errors[-1]
    for t in reversed(range(T - 1)):  # For each timestep in trajectory backwards
        E = td_errors[t] + (weights[t] * E)
        td_lambda_errors[t] = E

    return td_lambda_errors


def calculate_rbis_errors(lambd, td_errors, discount, dones, rhos):
    assert 0.0 <= lambd <= 1.0
    T = len(td_errors)

    zeros = np.zeros_like(td_errors)
    discount_products = zeros.copy()
    lambda_products = zeros.copy()
    betas = zeros.copy()

    rbis_errors = zeros.copy()
    for t in range(T):
        # Decay eligibilities
        discount_products *= discount
        lambda_products *= lambd
        betas = np.minimum(lambda_products, betas * rhos[t])

        # Increment eligibilities
        discount_products[t] = 1.0
        lambda_products[t] = 1.0
        betas[t] = 1.0

        # Accumulate updates
        rbis_errors += discount_products * betas * td_errors[t]

        # Reset if needed
        discount_products = np.where(dones[t], zeros, discount_products)
        lambda_products = np.where(dones[t], zeros, lambda_products)
        betas = np.where(dones[t], zeros, betas)

    return rbis_errors


def calculate_hybrid_returns(n, lambda_returns, rewards, bootstraps, discount, should_cut):
    assert n >= 1
    T = len(lambda_returns)
    assert n <= T  # TODO: Can we remove this assertion?

    if n == 1:
        return lambda_returns

    n_step_returns = np.copy(lambda_returns)
    for t in range(T):
        start = t
        end = min(t + n - 1, T - 1)

        G = lambda_returns[end]
        for i in reversed(range(start, end)):
            G = np.where(should_cut[i], bootstraps[i], G)  # Cut for off-policy corrections
            G = rewards[i] + (discount * G)
        n_step_returns[t] = G

    return n_step_returns


def calculate_harmonic_returns(rewards, bootstraps, discount, dones):
    T = len(rewards)

    all_n_step_returns = np.stack([
        calculate_n_step_returns(n, rewards, bootstraps, discount, dones)
        for n in range(1, T + 1)
    ])

    weights = np.zeros([T, T])
    for row in range(T):
        k = 0.0
        mask = 1.0
        for t in range(T - row):
            k += 1
            weights[row, t] = mask / k
            mask *= np.where(dones[row + t], 0.0, 1.0)

    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights * all_n_step_returns.T).sum(axis=-1)


def egreedy_probabilities(Q: np.ndarray, epsilon):
    greedy_actions = np.argmax(Q, axis=-1)
    num_actions = Q.shape[-1]
    not_greedy = 1.0 - one_hot(greedy_actions, num_actions)

    explore_probs = (epsilon / num_actions) * np.ones_like(not_greedy)
    greedy_probs = 1 - epsilon + explore_probs
    return np.where(not_greedy, explore_probs, greedy_probs)


def one_hot(x, num_classes):
    return np.identity(num_classes)[x]
