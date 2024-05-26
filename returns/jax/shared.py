import jax
from jax.lax import stop_gradient
import jax.numpy as jnp


def _select(array, index):
    return array[index]
vmap_select_axis1 = jax.vmap(_select, in_axes=[0, 0])


def calc_isrs(t_probs, b_probs):
    return stop_gradient(t_probs / b_probs)


def calc_dones(terms, truncs):
    # Marks episode boundaries, either due to termination or truncation.
    # Because we do not store the next obs when an episode ends, this occurs
    # one time step earlier for truncation than for termination, as the former
    # must be able to bootstrap from a valid obs.
    return jnp.logical_or(terms[:-1], truncs[1:])


def calc_td_errors(values, bootstraps, rewards, terms, discount):
    bootstraps = jnp.where(terms[:-1], 0.0, bootstraps[1:])
    return rewards[:-1] + (discount * bootstraps) - values[:-1]


def calc_nstep_returns(n, values, bootstraps, rewards, terms, truncs, discount, factors=None):
    td_errors = calc_td_errors(values, bootstraps, rewards, terms, discount)

    dones = calc_dones(terms, truncs)
    if factors is None:
        factors = jnp.ones_like(rewards)
    factors = jnp.where(dones, 0.0, discount * factors[1:])

    nstep_errors = jnp.empty_like(td_errors)
    L = len(nstep_errors)
    for t in range(L):
        nstep_errors = nstep_errors.at[t].set(
            _calc_nstep_error(n, t, td_errors, factors)
        )

    nstep_returns = values[:-1] + nstep_errors
    where_safe = jnp.logical_not(truncs[:-1])  # Can't train on samples where truncated=True
    return nstep_returns, where_safe


def _calc_nstep_error(n, start, td_errors, factors):
    assert n >= 1
    L = len(td_errors)
    end = min(start + n, L) - 1

    e = td_errors[end]
    for i in reversed(range(start, end)):
        e = td_errors[i] + (factors[i] * e)
    return e


def calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, factors=None):
    assert 0.0 <= lambd <= 1.0
    td_errors = calc_td_errors(values, bootstraps, rewards, terminateds, discount)
    dones = jnp.logical_or(terminateds, truncateds)  # End of episode, regardless of reason

    if lambd == 0.0:
        return values[:-1] + td_errors

    if factors is None:
        factors = jnp.ones_like(rewards)
    weights = jnp.where(dones[:-1], 0.0, discount * lambd * factors[1:])

    lambda_errors = jnp.copy(td_errors)
    e = lambda_errors[-1]
    L = len(lambda_errors)
    for t in reversed(range(L - 1)):  # For each time step in trajectory, backwards
        e = td_errors[t] + (weights[t] * e)
        lambda_errors = lambda_errors.at[t].set(e)

    return values[:-1] + lambda_errors


def calc_rbis_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, factors):
    assert 0.0 <= lambd <= 1.0
    td_errors = calc_td_errors(values, bootstraps, rewards, terminateds, discount)
    dones = jnp.logical_or(terminateds, truncateds)  # End of episode, regardless of reason

    zeros = jnp.zeros_like(td_errors)
    discount_products = zeros.copy()
    lambda_products = zeros.copy()
    betas = zeros.copy()

    rbis_errors = zeros.copy()
    L = len(rbis_errors)
    for t in range(L):
        # Decay eligibilities
        discount_products *= discount
        lambda_products *= lambd
        betas = jnp.minimum(lambda_products, betas * factors[t])

        # Increment eligibilities
        discount_products = discount_products.at[t].set(1)
        lambda_products = lambda_products.at[t].set(1)
        betas = betas.at[t].set(1)

        # Accumulate updates
        rbis_errors += discount_products * betas * td_errors[t]

        # Reset if needed
        discount_products = jnp.where(dones[t], zeros, discount_products)
        lambda_products = jnp.where(dones[t], zeros, lambda_products)
        betas = jnp.where(dones[t], zeros, betas)

    return values[:-1] + rbis_errors
