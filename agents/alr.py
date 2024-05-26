import itertools
import math

import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import numpy as np

from agents.dqn import DQN, fast_nstep_return, huber_loss


class ALR(DQN):
    """DQN with approximate lambda-return from averaged n-step returns"""

    def _make_estimator(self, name):
        self.estimator = name  # For backwards compatibility

    def _define_update(self):
        discount = self.discount
        est = self.estimator

        prefix, effective_n = est.split('-')
        effective_n = int(effective_n)

        if prefix == 'nstep':
            n1 = n2 = effective_n
            w = 1.0  # Value doesn't matter, but set to 1 to optimize return calculation below

        elif prefix == 'pilar':
            (n1, n2, w), error = best_approximation(effective_n, discount)
            print("w={} --> error={}".format(w, error))

        elif prefix == 'pilar1':
            n1 = 1
            n2 = effective_n + 1

        elif prefix == 'pilar2':
            n1 = effective_n - 1
            n2 = effective_n + 1

        elif prefix == 'pilar3':
            n1 = effective_n - 1
            n2 = effective_n + 2

        elif prefix == 'pilar4':
            n1 = math.ceil(effective_n / 2)
            n2 = math.floor(3 * effective_n / 2)

        else:
            raise ValueError(f"unsuppported return estimator '{est}'")

        assert 1 <= n1 <= n2
        self.traj_len = n2 + 1

        if prefix != 'nstep':
            assert discount > 0
            if discount < 1:
                w = (pow(discount, effective_n) - pow(discount, n1)) / (pow(discount, n2) - pow(discount, n1))
            else:
                w = (effective_n - n1) / (n2 - n1)

        print("n={} --> (n1, n2)={}, w={}".format(effective_n, (n1, n2), w))
        cr1 = (1-w) * pow(discount, n1) + w * pow(discount, n2)
        cr2 = pow(discount, effective_n)
        print("testing if {} ~= {}".format(cr1, cr2))
        assert np.allclose(cr1, cr2), "contraction rate check failed"

        def trajectory_loss(params, target_params, obs, actions, rewards, terminateds, truncateds):
            # Just need to compute the first Q-value of the sequence with main parameters
            Q_main = self.q_values(params, obs[0, None])
            q_main_taken = Q_main[0, actions[0]]

            value_func = lambda s: jnp.max(self.q_values(target_params, s), axis=-1)
            nstep_returns = lambda n: fast_nstep_return(n, value_func, obs, rewards, terminateds, truncateds, discount)

            if w == 0.0:
                G, where_safe = nstep_returns(n1)
            elif w == 1.0:
                G, where_safe = nstep_returns(n2)
            else:
                G1, _ = nstep_returns(n1)
                G2, where_safe = nstep_returns(n2)
                G = (1-w) * G1 + w * G2

            error = stop_gradient(G) - q_main_taken
            loss = {
                'mse': 0.5 * jnp.square(error),
                'huber': huber_loss(error),
            }[self.loss]
            return jnp.where(where_safe, loss, 0.0)

        vmap_trajectory_loss = jax.vmap(trajectory_loss, in_axes=[None, None, 0, 0, 0, 0, 0])

        @jax.jit
        def update(opt_state, target_params, minibatch, t):
            def loss(params):
                losses = vmap_trajectory_loss(params, target_params, *minibatch)
                return jnp.mean(losses)

            params = self.get_params(opt_state)
            step = jax.grad(loss)(params)
            opt_state = self.opt_update(t, step, opt_state)
            return opt_state

        self.update = update


def best_approximation(effective_n, discount):
    assert effective_n >= 1
    assert 0.0 < discount < 1.0
    lambd = (1 - pow(discount, effective_n - 1)) / (1 - pow(discount, effective_n))

    def error_func(n1, n2):
        N = 10_000  # Number of terms in approximation
        pilar_weight, w = get_pilar_weight_func(effective_n, discount, n1, n2)
        error = max([abs(pilar_weight(i) - pow(discount * lambd, i)) for i in range(N + 1)])
        return error, w

    best_values = None
    best_error = float('inf')

    for n1 in range(1, math.floor(effective_n) + 1):
        prev_error = float('inf')

        for n2 in itertools.count(start=math.floor(effective_n) + 1):
            error, w = error_func(n1, n2)

            if error < best_error:
                best_values = (n1, n2, w)
                best_error = error

            if error >= prev_error:
                break
            prev_error = error

    # Sanity check: make sure contraction rates match
    cr = (1-w) * pow(discount, n1) + w * pow(discount, n2)
    expected_cr = pow(discount, effective_n)
    assert np.allclose(cr, expected_cr), f"contraction rate sanity check failed: {cr} != {expected_cr}"

    return best_values, best_error


def get_pilar_weight_func(effective_n, discount, n1, n2):
    assert n1 <= effective_n < n2
    assert 0.0 < discount < 1.0
    w = (discount**n1 - discount**effective_n) / (discount**n1 - discount**n2)

    def pilar_weight(i):
        if i < n1:
            return pow(discount, i)
        if i < n2:
            return w * pow(discount, i)
        return 0.0

    return pilar_weight, w
