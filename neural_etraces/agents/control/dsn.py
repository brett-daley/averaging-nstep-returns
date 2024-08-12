import jax
from jax.lax import stop_gradient
import jax.numpy as jnp

from agents.control.dqn import DQN, huber_loss
import returns.jax as returns


class DSN(DQN):
    """Deep Sarsa Network with forward-view returns"""

    def _make_estimator(self, name):
        self.est = returns.sarsa.get_estimator(name)
        self.traj_len = self.est.traj_len(self.batch_len)

    def _define_update(self):
        batch_len = self.batch_len
        discount = self.discount
        est = self.est

        def trajectory_loss(params, target_params, obs, actions, rewards, terminateds, truncateds, b_probs):
            Q_main = self.q_values(params, obs)
            q_main_taken = returns.vmap_select_axis1(Q_main, actions)

            Q_targ = self.q_values(target_params, obs)
            q_targ_taken = returns.vmap_select_axis1(Q_targ, actions)

            t_probs = returns.vmap_select_axis1(self.distr(Q_targ), actions)

            g_targ, where_safe = est.calc_returns(q_targ_taken, q_targ_taken, rewards, terminateds, truncateds, discount, t_probs, b_probs)

            errors = stop_gradient(g_targ) - q_main_taken[:-1]  # Make relative to main network
            losses = {
                'mse': 0.5 * jnp.square(errors),
                'huber': huber_loss(errors),
            }[self.loss]
            losses = jnp.where(where_safe, losses, 0.0)
            return losses[:batch_len]

        vmap_trajectory_loss = jax.vmap(trajectory_loss, in_axes=[None, None, 0, 0, 0, 0, 0, 0])

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

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, b_prob):
        self.replay_memory.save(obs, action, reward, terminated, truncated, b_prob)
        self.update_target_network()

        if self.t <= self.prepop:
            return

        if self.train_period == 1 or (self.t % self.train_period) == 1:
            minibatch = self.replay_memory.sample_trajectories(self.batch_size, length=self.traj_len)
            self.opt_state = self.update(self.opt_state, self.target_params, minibatch, self.train_iterations)
            self.train_iterations += 1
