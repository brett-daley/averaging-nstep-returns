from abc import ABC, abstractmethod
from collections import namedtuple
import argparse
import itertools

import gymnasium as gym
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import numpy as np

from averaging_nstep_returns import cmdline, extractors, optimizers, schedules
from averaging_nstep_returns.data_structures import ReplayMemory
from averaging_nstep_returns.tasks import envs
from averaging_nstep_returns.utils import jax_device_context, use_deterministic_gpu_ops
from averaging_nstep_returns.extractors import vector_ops as vect
import averaging_nstep_returns.returns.jax as returns
from averaging_nstep_returns.utils import egreedy


class Agent(ABC):
    def __init__(self, obs_space, action_space, rng_seed: int):
        self.obs_space = obs_space
        self.action_space = action_space
        self.rng_seed = rng_seed
        self.np_random = np.random.default_rng(rng_seed)

    @abstractmethod
    def reinforce(self, obs, action, next_obs, reward, done, info={}):
        raise NotImplementedError


class ControlAgent(Agent):
    def __init__(self, obs_space, action_space, seed: int, discount: float):
        super().__init__(obs_space, action_space, seed)
        assert 0.0 <= discount <= 1.0
        self.discount = discount

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError


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


def main(**kwargs):  # Hook for automation
    kwargs = cmdline.insert_defaults(kwargs)
    cmdline.assert_not_none(kwargs)

    force_cpu = kwargs.pop('cpu', False)
    if not force_cpu:
        use_deterministic_gpu_ops()

    with jax_device_context(force_cpu):
        return run(**kwargs)


def run(env: str, agent: str, discount: float, duration: float, seed: int, verbose: bool = False, **agent_kwargs):
    duration = int(duration)
    assert duration > 0
    assert 0.0 <= discount <= 1.0

    # Make environment
    env = envs.make(env)
    env.action_space.seed(seed)

    # Make agent
    agent = ALR(env.observation_space, env.action_space, seed, discount, **agent_kwargs)

    # Start training

    time_periods = 0
    period_start = env.time()
    period_length_minutes = 15

    obs, _ = env.reset(seed=seed)  # Pass seed to initialized RNG

    performance = np.zeros(duration + 1)
    i = 0

    for t in itertools.count(start=1):
        action, b_prob = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.reinforce(obs, action, next_obs, reward, terminated, truncated, b_prob)

        if env.is_done():
            avg_undisc_return = np.mean(env.get_episode_returns()[-100:])

            while i <= t:
                performance[i] = avg_undisc_return

                if (i % 500) == 0 and verbose:
                    episodes = env.get_total_episodes()
                    last_episode_return = env.get_episode_returns()[-1]
                    print(f"{env.time():.2f}s  t={i}  ep={episodes}  {last_episode_return} (avg: {avg_undisc_return:.2f})")

                if i == duration:
                    return performance
                i += 1

            next_obs, _ = env.reset()

        obs = next_obs

        # Periodic logging even when verbose=False (for time estimation on Compute Canada)
        minutes = (env.time() - period_start) / 60
        if minutes >= period_length_minutes:
            time_periods += 1
            percent_complete = round(100 * t / duration, 1)
            print(f"Approximately {round(time_periods * period_length_minutes, 1)} minutes elapsed; "
                  f"{t}/{duration} timesteps ({percent_complete}%) completed")
            period_start = env.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--agent', type=str, default='ALR')
    parser.add_argument('--defaults', type=str)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--duration', type=float, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    kwargs = cmdline.parse_kwargs(parser)
    main(**kwargs)
