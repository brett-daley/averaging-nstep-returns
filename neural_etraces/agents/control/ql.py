from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp

from agents import ControlAgent
from agents.control.dqn import argmax, dueling_layer
import extractors
from extractors import vector_ops as vect
import optimizers
import schedules


class QL(ControlAgent):
    """Q(λ) without experience replay. Watkins' Q(λ) if cut_traces=True, Peng's Q(λ) if cut_traces=False."""
    Parameters = namedtuple('Parameters', ['theta', 'w', 'b'])

    def __init__(self, observation_space, action_space, seed, discount, extractor='none', opt='adam', lr=3e-4,
                 epsilon=0.05, target_period=1, dueling='none', lambd=0.0, cut_traces=True, tnc=True, al=0.0, **kwargs):
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)
        assert lr > 0.0
        assert 0.0 <= lambd <= 1.0

        super().__init__(observation_space, action_space, seed, discount)
        self.extractor = extractor = extractors.make(extractor)
        self.opt_cls = getattr(optimizers, opt)
        self.lr = lr
        self.epsilon_schedule = schedules.make(epsilon)
        self.target_period = target_period
        self.dueling = dueling
        self.lambd = lambd
        assert isinstance(cut_traces, bool)
        self.cut_traces = cut_traces
        assert isinstance(tnc, bool)
        self.tnc = tnc
        assert 0.0 <= al < 1.0
        self.al = al

        input_shape = observation_space.shape
        prng_key = jax.random.PRNGKey(seed)

        theta, features, prng_key = extractor.generate_parameters(input_shape, prng_key)
        w = jnp.zeros([features, action_space.n + 1])
        b = jnp.zeros([1, w.shape[1]])
        self.init_params = self.Parameters(theta, w, b)
        self.z = vect.zeros_like(self.init_params)

        self._define_forward(extractor)
        self._define_update()
        self.t = 0

        # Set up target network
        self.target_params = self.params

    def _define_forward(self, extractor):
        # Make optimizer
        self.opt_init, self.opt_update, self.get_params = self.opt_cls(self.lr)
        self.opt_state = self.opt_init(self.init_params)

        def features(params, obs):
            return extractor.forward(params.theta, obs)
        self.features = features

        def qvalues(params, obs):
            x = features(params, obs)
            y = x.dot(params.w) + params.b  # Linear layer
            Q = dueling_layer(y, mode=self.dueling)
            return Q
        self.qvalues = qvalues
        self.jit_qvalues = jax.jit(qvalues)
        self.v = lambda params, obs: jnp.max(self.qvalues(params, obs), axis=-1)

        def get_qvalue(params, obs, action):
            q = qvalues(params, obs)
            return q[0, action]
        self.get_qvalue = get_qvalue

    def _define_update(self):
        discount = self.discount
        lambd = self.lambd
        cut_traces = self.cut_traces
        using_target_net = (self.target_period > 1)
        use_tnc = self.tnc and using_target_net  # Target-network corrections
        al = self.al  # Advantage learning coefficient
        if al > 0.0 and lambd > 0.0:
            raise NotImplementedError

        @partial(jax.jit, static_argnames=['terminated', 'truncated'])
        def update(opt_state, z, target_params, obs, action, next_obs, reward, terminated, truncated, t):
            params = self.get_params(opt_state)

            q_main, grads = jax.value_and_grad(self.get_qvalue)(params, obs, action)

            v_main = None
            if al > 0.0:
                v_main = self.v(params, obs)

            # Calculate TD error
            td_error = reward - q_main
            if not terminated:
                # Bootstrap
                next_v_targ = self.v(target_params, next_obs)
                td_error += discount * next_v_targ

            if al > 0.0:  # Advantage learning
                td_error -= al * (v_main - q_main)

            if lambd != 0.0:  # Q(λ)
                z = vect.scale(discount * lambd, z)  # Decay traces

                # Calculate telescoping TD error
                tele_td_error = reward
                if not terminated:
                    # Bootstrap
                    tele_td_error += discount * next_v_targ

                if use_tnc:
                    # Target-network corrections: subtract by the target parameters' estimate
                    v_targ = self.v(target_params, obs)
                    tele_td_error -= v_targ
                else:
                    # No corrections: subtract by the main parameters' estimate
                    v_main = self.v(params, obs)
                    tele_td_error -= v_main

                step = vect.add(
                    vect.scale(tele_td_error, z),
                    vect.scale(td_error, grads)
                )

                z = vect.add(z, grads)  # Spike traces after update

            else:  # Q(0), eligibility traces not used
                step = vect.scale(td_error, grads)

            step = vect.scale(-1, step)  # Flip sign for gradient *descent*
            opt_state = self.opt_update(t, step, opt_state)

            # Reset traces if needed
            if terminated or truncated:
                z = vect.zeros_like(z)
            elif cut_traces:  # True -> Watkins' Q(λ), False -> Peng's Q(λ).
                if v_main is None:
                    v_main = self.v(params, obs)
                not_greedy = (q_main != v_main)
                z = vect.conditional_zeros_like(z, not_greedy)

            return opt_state, z

        self.update = update

    @property
    def params(self):
        return self.get_params(self.opt_state)

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, b_prob):
        self.update_target_network()
        minibatch = (obs[None], action, next_obs[None], reward, terminated, truncated)
        self.opt_state, self.z = self.update(self.opt_state, self.z, self.target_params, *minibatch, self.t)

    def update_target_network(self):
        if self.target_period == 1:
            self.target_params = self.params
            return

        if (self.t % self.target_period) == 1:
            self.target_params = vect.copy(self.params)

    def act(self, obs):
        self.t += 1
        return self._sample_egreedy(obs)

    def _sample_egreedy(self, obs):
        epsilon = self.epsilon_schedule(self.t)
        assert 0.0 <= epsilon <= 1.0

        if self.np_random.random() <= epsilon:
            prob = epsilon / self.action_space.n
            return self.action_space.sample(), prob

        q = self.jit_q_values(self.params, obs[None])[0]  # Add/remove batch dimension

        prob = 1 - epsilon + (epsilon / self.action_space.n)
        return argmax(q), prob
