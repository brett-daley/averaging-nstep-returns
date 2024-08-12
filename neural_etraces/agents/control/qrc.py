from collections import namedtuple

import jax
from jax.lax import stop_gradient
import jax.numpy as jnp

from agents.control.dqn import DQN, dueling_layer
import extractors
import returns.jax as returns


class QRC(DQN):
    """Q-Learning with Regularized Corrections and forward-view returns"""
    Parameters = namedtuple('Parameters', ['theta', 'w', 'b', 'wh', 'bh'])

    def _make_network(self, extractor, seed):
        self.extractor = extractor = extractors.make(extractor)
        input_shape = self.obs_space.shape
        prng_key = jax.random.PRNGKey(seed)
        theta, features, prng_key = extractor.generate_parameters(input_shape, prng_key)
        w = jnp.zeros([features, self.action_space.n + 1])
        b = jnp.zeros(w.shape[1])
        self.init_params = self.Parameters(theta, w, b, w.copy(), b.copy())

    def _define_forward(self):
        super()._define_forward()

        def h_values(params, obs):
            x = self.features(params, obs)
            # Don't backprop gradients through the feature extractor here
            y = stop_gradient(x).dot(params.wh) + params.bh  # Linear layer
            return dueling_layer(y, mode=self.dueling)
        self.h_values = h_values

    def _define_update(self):
        batch_len = self.batch_len
        discount = self.discount
        est = self.est
        loss = self.loss
        assert loss == 'mse'

        def trajectory_loss(params, obs, actions, rewards, terminateds, truncateds):
            Q = self.q_values(params, obs)
            q_taken = returns.vmap_select_axis1(Q, actions)
            v = jnp.max(Q, axis=-1)
            where_greedy = (q_taken == v)

            g, where_safe = est.calc_returns(v, v, rewards, terminateds, truncateds, discount, where_greedy)

            H = self.h_values(params, obs)
            h = returns.vmap_select_axis1(H, actions)

            losses = tdc_loss(g, q_taken[:-1], h[:-1])
            losses = jnp.where(where_safe, losses, 0.0)
            return losses[:batch_len]

        vmap_trajectory_loss = jax.vmap(trajectory_loss, in_axes=[None, 0, 0, 0, 0, 0])

        @jax.jit
        def update(opt_state, target_params, minibatch, t):
            def loss(params):
                losses = vmap_trajectory_loss(params, *minibatch)
                return jnp.mean(losses)

            params = self.get_params(opt_state)
            step = jax.grad(loss)(params)
            step = self._regularized_corrections(params, step)
            opt_state = self.opt_update(t, step, opt_state)
            return opt_state

        self.update = update

    def _regularized_corrections(self, params, step):
        beta = 1.0  # TODO: Don't hardcode
        # Normally we subtract the L2 penalty, but here we add
        # because the sign gets flipped by the optimizer
        return step._replace(wh=(step.wh + beta * params.wh))


def tdc_loss(returns, values, h):
    errors = returns - values
    v_loss = stop_gradient(h) * returns - stop_gradient(errors) * values
    h_loss = stop_gradient(h - errors) * h
    return v_loss + h_loss
