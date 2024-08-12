from functools import partial

from returns import Estimator
from returns.jax import esarsa, shared


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

    elif name.startswith('rbis-'):
        _, lambd = name.split('-')
        est.calc_returns = partial(calc_rbis_returns, float(lambd))

    else:
        raise ValueError(f"unknown return estimator '{name}'")

    return est


calc_nstep_returns = esarsa.calc_nstep_returns


def calc_opc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    returns = esarsa.calc_opc_nstep_returns(n, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs)
    return apply_sarsa_corrections(values, returns, t_probs, b_probs)


calc_lambda_returns = esarsa.calc_lambda_returns


def calc_opc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    returns = esarsa.calc_opc_lambda_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs)
    return apply_sarsa_corrections(values, returns, t_probs, b_probs)


def calc_rbis_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, t_probs, b_probs):
    isrs = shared.calc_isrs(t_probs, b_probs)
    returns = shared.calc_rbis_returns(lambd, values, bootstraps, rewards, terminateds, truncateds, discount, isrs)
    return apply_sarsa_corrections(values, returns, t_probs, b_probs)


def apply_sarsa_corrections(values, returns, t_probs, b_probs):
    errors = returns - values[:-1]
    # Since Sarsa uses sample backups, we must multiply by the ISRs
    isrs = shared.calc_isrs(t_probs, b_probs)
    return values[:-1] + (isrs[1:] * errors)
