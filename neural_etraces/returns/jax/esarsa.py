from functools import partial

import jax
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

    elif name.startswith('retrace-nstep-'):
        _, _, n = name.split('-')
        n = int(n)
        est.traj_len = lambda batch_len: batch_len + n
        est.calc_returns = partial(calc_retrace_nstep_returns, n)

    elif name.startswith('lambda-'):
        _, lambd = name.split('-')
        est.calc_returns = partial(calc_lambda_returns, float(lambd))

    elif name.startswith('opc-lambda-'):
        _, _, lambd = name.split('-')
        est.calc_returns = partial(calc_opc_lambda_returns, float(lambd))

    elif name.startswith('rbis-'):
        _, lambd = name.split('-')
        est.calc_returns = partial(calc_rbis_returns, float(lambd))

    else:
        raise ValueError(f"unknown return estimator '{name}'")

    return est


def calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs=None, b_probs=None):
    return shared.calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount)


def calc_opc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    isrs = shared.calc_isrs(t_probs, b_probs)
    return shared.calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, isrs)


def calc_retrace_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    isrs = shared.calc_isrs(t_probs, b_probs)
    return shared.calc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, factors=jnp.minimum(1.0, isrs))


def calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs=None, b_probs=None):
    return shared.calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount)


def calc_opc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    isrs = shared.calc_isrs(t_probs, b_probs)
    return shared.calc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, isrs)


def calc_rbis_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    isrs = shared.calc_isrs(t_probs, b_probs)
    return shared.calc_rbis_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, isrs)
