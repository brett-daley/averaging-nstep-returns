import jax
import jax.numpy as jnp

from agents.control.dqn import fast_nstep_return
from agents.prediction import TD
from data_structures import ReplayMemory


class TDReplay(TD):
    def __init__(self, observation_space, action_space, seed, discount, train_period=4,
                 prepop=50_000, rmem_size=500_000, batch_size=32, **kwargs):
        super().__init__(observation_space, action_space, seed, discount, **kwargs)
        assert train_period >= 1
        self.train_period = train_period
        self.prepop = prepop
        assert batch_size >= 1
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(rmem_size, seed)

    def _define_update(self):
        discount = self.discount

        @jax.jit
        def trajectory_loss(params, target_params, obs, rewards, terminateds, truncateds):
            # Just need to compute the first Q-value of the sequence with main parameters
            v = self.value_function(params, obs[0, None])

            dones = jnp.logical_or(terminateds, truncateds)  # End of episode, regardless of reason

            value_func = lambda s: self.value_function(target_params, s)
            n = 1
            G = fast_nstep_return(n, value_func, obs, rewards[:-1], terminateds[:-1], dones[:-1], discount)
            G = jax.lax.stop_gradient(G)

            return 0.5 * jnp.square(G - v)

        vmap_trajectory_loss = jax.vmap(trajectory_loss, in_axes=[None, None, 0, 0, 0, 0])

        @jax.jit
        def update(opt_state, target_params, obs, actions, rewards, terminateds, truncateds, t):
            def loss(params):
                losses = vmap_trajectory_loss(params, target_params, obs, rewards, terminateds, truncateds)
                return jnp.mean(losses)

            params = self.get_params(opt_state)
            step = jax.grad(loss)(params)
            opt_state = self.opt_update(t, step, opt_state)
            return opt_state

        self.update = update

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated):
        self.t += 1
        self.replay_memory.save(obs, action, reward, terminated, truncated, b_prob=0.0)
        self.update_target_network()

        if self.t <= self.prepop:
            return

        if self.train_period == 1 or (self.t % self.train_period) == 1:
            minibatch = self.replay_memory.sample_trajectories(self.batch_size, length=2)
            minibatch = minibatch[:-1]  # Slice off behavior probabilities
            self.opt_state = self.update(self.opt_state, self.target_params, *minibatch, self.t)
