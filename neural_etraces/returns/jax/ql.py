from functools import partial

import jax.numpy as jnp

from returns import Estimator
from returns.jax import shared


def get_estimator(name: str):
    est = Estimator()

    if name.startswith('nstep-'):
        _, n = name.split('-')
        n = int(n)
        est.traj_len = lambda batch_len: batch_len + n
        est.calc_returns = partial(calc_nstep_returns, n)

    elif name.startswith('opc-nstep-'):
        _, _, n = name.split('-')
        n = int(n)
        est.traj_len = lambda batch_len: batch_len + n
        est.calc_returns = partial(calc_opc_nstep_returns, n)

    elif name.startswith('lambda-'):
        _, lambd = name.split('-')
        est.calc_returns = partial(calc_lambda_returns, float(lambd))

    elif name.startswith('opc-lambda-'):
        _, _, lambd = name.split('-')
        est.calc_returns = partial(calc_opc_lambda_returns, float(lambd))

    else:
        raise ValueError(f"unknown return estimator '{name}'")

    return est


def calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, where_greedy=None):
    return shared.calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount)


def calc_opc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, where_greedy):
    trace_cuts = jnp.where(where_greedy, 1.0, 0.0)
    return shared.calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, trace_cuts)


def calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, where_greedy=None):
    return shared.calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount)


def calc_opc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, where_greedy):
    trace_cuts = jnp.where(where_greedy, 1.0, 0.0)
    return shared.calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, trace_cuts)
