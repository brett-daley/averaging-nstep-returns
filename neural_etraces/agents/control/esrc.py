from operator import itemgetter

import jax
from jax.lax import stop_gradient
import jax.numpy as jnp

from agents.control.dsn import DSN
from agents.control.qrc import QRC, tdc_loss
import returns.jax as returns


class ESRC(QRC):
    """Expected Sarsa with Regularized Corrections and forward-view returns"""

    def _make_estimator(self, name):
        self.est = returns.esarsa.get_estimator(name)
        self.traj_len = self.est.traj_len(self.batch_len)

    def _define_update(self):
        batch_len = self.batch_len
        discount = self.discount
        est = self.est
        loss = self.loss
        assert loss == 'mse'

        def trajectory_loss(params, obs, actions, rewards, terminateds, truncateds, b_probs):
            Q = self.q_values(params, obs)
            q_taken = returns.vmap_select_axis1(Q, actions)

            action_distr = self.distr(Q)
            v = jnp.sum(action_distr * Q, axis=1)

            t_probs = returns.vmap_select_axis1(action_distr, actions)

            g, where_safe = est.calc_returns(q_taken, v, rewards, terminateds, truncateds, discount, t_probs, b_probs)

            H = self.h_values(params, obs)
            h = returns.vmap_select_axis1(H, actions)

            losses = tdc_loss(g, q_taken[:-1], h[:-1])
            losses = jnp.where(where_safe, losses, 0.0)
            return losses[:batch_len]

        vmap_trajectory_loss = jax.vmap(trajectory_loss, in_axes=[None, 0, 0, 0, 0, 0, 0])

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

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, b_prob):
        return DSN.reinforce(self, obs, action, next_obs, reward, terminated, truncated, b_prob)
