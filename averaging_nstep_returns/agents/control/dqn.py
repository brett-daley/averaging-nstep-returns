from collections import namedtuple

import gymnasium as gym
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import numpy as np

from agents import ControlAgent
from data_structures import ReplayMemory
import extractors
from extractors import vector_ops as vect
import optimizers
import returns.jax as returns
import schedules
from utils import egreedy


class DQN(ControlAgent):
    """Deep Q-Network with forward-view returns"""
    Parameters = namedtuple('Parameters', ['theta', 'w', 'b'])

    def __init__(self, observation_space, action_space, seed, discount, extractor='none', opt='adam', lr=3e-4, train_period=4,
                 epsilon=0.05, prepop=50_000, target_period=1, dueling='none', rmem_size=500_000,
                 est='nstep-1', batch_size=64, batch_len=1, loss='mse'):
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)

        super().__init__(observation_space, action_space, seed, discount)
        assert train_period >= 1
        self.train_period = train_period
        self.epsilon_schedule = schedules.make(epsilon)
        self.prepop = prepop

        self.target_period = target_period
        self.dueling = dueling

        self.replay_memory = ReplayMemory(rmem_size, seed)
        assert batch_size >= 1
        self.batch_size = batch_size
        assert batch_len >= 1
        self.batch_len = batch_len
        self.loss = loss

        self._make_network(extractor, seed)
        self._make_estimator(est)
        self._make_optimizer(opt, lr)
        self._define_forward()
        self._define_update()
        self.t = 0
        self.train_iterations = 0

        # Debug info
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print("Parameters =", param_count)
        self._print_q = False

    def _make_network(self, extractor, seed):
        self.extractor = extractor = extractors.make(extractor)
        input_shape = self.obs_space.shape
        prng_key = jax.random.PRNGKey(seed)
        theta, features, prng_key = extractor.generate_parameters(input_shape, prng_key)
        w = jnp.zeros([features, self.action_space.n + 1])
        b = jnp.zeros(w.shape[1])
        self.init_params = self.Parameters(theta, w, b)

    def _make_estimator(self, name):
        self.est = returns.ql.get_estimator(name)
        self.traj_len = self.est.traj_len(self.batch_len)

    def _make_optimizer(self, opt, lr):
        assert lr > 0.0
        self.lr = lr
        opt_cls = getattr(optimizers, opt)
        self.opt_init, self.opt_update, self.get_params = opt_cls(lr)
        self.opt_state = self.opt_init(self.init_params)

    def _define_forward(self):
        def features(params, obs):
            return self.extractor.forward(params.theta, obs)
        self.features = features

        def q_values(params, obs):
            x = features(params, obs)
            y = x.dot(params.w) + params.b  # Linear layer
            Q = dueling_layer(y, mode=self.dueling)
            return Q
        self.q_values = q_values
        self.jit_q_values = jax.jit(q_values)

    def _define_update(self):
        batch_len = self.batch_len
        discount = self.discount
        est = self.est

        def trajectory_loss(params, target_params, obs, actions, rewards, terminateds, truncateds):
            Q_main = self.q_values(params, obs)
            q_main_taken = returns.vmap_select_axis1(Q_main, actions)

            v_targ = self.dqn_target(Q_main, Q_targ=self.q_values(target_params, obs))
            where_greedy = (q_main_taken == v_targ)

            g_targ, where_safe = est.calc_returns(v_targ, v_targ, rewards, terminateds, truncateds, discount, where_greedy)

            errors = stop_gradient(g_targ) - q_main_taken[:-1]  # Make relative to main network
            losses = {
                'mse': 0.5 * jnp.square(errors),
                'huber': huber_loss(errors),
            }[self.loss]
            losses = jnp.where(where_safe, losses, 0.0)
            return losses[:batch_len]

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

    @staticmethod
    def dqn_target(Q_main, Q_targ):
        return jnp.max(Q_targ, axis=-1)

    @property
    def params(self):
        return self.get_params(self.opt_state)

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, b_prob):
        self.replay_memory.save(obs, action, reward, terminated, truncated, b_prob)
        self.update_target_network()

        if self.t <= self.prepop:
            return

        if self.train_period == 1 or (self.t % self.train_period) == 1:
            minibatch = self.replay_memory.sample_trajectories(self.batch_size, length=self.traj_len)
            minibatch = minibatch[:-1]  # Slice off behavior probabilities
            self.opt_state = self.update(self.opt_state, self.target_params, minibatch, self.train_iterations)
            self.train_iterations += 1

    def update_target_network(self):
        if self.target_period == 1:
            self.target_params = self.params
            return

        if (self.t % self.target_period) == 1:
            self.target_params = vect.copy(self.params)

    def act(self, obs):
        self.t += 1

        self._print_q = self._print_q or (self.t % 1_000 == 1)

        epsilon = self._epsilon()
        assert 0.0 <= epsilon <= 1.0

        if self.np_random.random() <= epsilon:
            prob = epsilon / self.action_space.n
            return self.action_space.sample(), prob

        q = self.jit_q_values(self.params, obs[None])[0]  # Add/remove batch dimension

        if self._print_q:
            print(self.t, q, f"ε={epsilon}")
            self._print_q = False

        prob = 1 - epsilon + (epsilon / self.action_space.n)
        return argmax(q), prob

    def _epsilon(self):
        if self.t < self.prepop:
            return 1.0
        return self.epsilon_schedule(self.t - self.prepop)

    def distr(self, Q):
        return egreedy(Q, self._epsilon())


class DDQN(DQN):
    @staticmethod
    def dqn_target(Q_main, Q_targ):
        argmax_actions = jnp.argmax(Q_main, axis=-1)
        return returns.vmap_select_axis1(Q_targ, argmax_actions)


def argmax(q):
    assert not np.isnan(q).all(), "cannot have NaN inputs"
    return np.argmax(q).item()


def dueling_layer(values: jnp.array, mode: str):
    assert values.ndim == 2
    if mode == 'none':
        return values[:, 1:]

    V = values[:, 0, None]
    A = values[:, 1:]

    reduce = {
        'max': jnp.max,
        'mean': jnp.mean,
    }[mode]
    A_ident = reduce(A, axis=-1, keepdims=True)
    return V + A - A_ident


def huber_loss(x):
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x < 1.0,
        0.5 * jnp.square(x),
        abs_x - 0.5
    )


def fast_nstep_return(n, value_func, obs, rewards, terms, truncs, discount):
    def bootstrap(i):
        v = value_func(obs[i+1, None])
        return jnp.where(terms[i], 0, v)

    dones = returns.shared.calc_dones(terms, truncs)

    bs_index = n - 1
    G = 0.0
    for i in reversed(range(n)):
        bs_index = jnp.where(dones[i], i, bs_index)
        G = rewards[i] + jnp.where(dones[i], 0.0, discount * G)

    G += jnp.power(discount, bs_index + 1) * bootstrap(bs_index)
    where_safe = jnp.logical_not(truncs[0])
    return G, where_safe