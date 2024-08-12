from collections import namedtuple
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp

from agents import PredictionAgent
import extractors
from extractors import vector_ops as vect
import optimizers


class TD(PredictionAgent):
    Parameters = namedtuple('Parameters', ['theta', 'w'])

    def __init__(self, observation_space, action_space, seed, discount, extractor='none', opt='adam', lr=3e-4,
                 lambd=0.0, target_period=1, **kwargs):
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)
        assert lr > 0.0
        assert 0.0 <= lambd <= 1.0
        self.lambd = lambd

        super().__init__(observation_space, action_space, seed, discount)
        self.extractor = extractor = extractors.make(extractor)
        self.opt_cls = getattr(optimizers, opt)
        self.lr = lr
        assert target_period >= 1
        self.target_period = target_period

        input_shape = observation_space.shape
        prng_key = jax.random.PRNGKey(seed)

        theta, features, prng_key = extractor.generate_parameters(input_shape, prng_key)
        w = jnp.zeros(features)

        self.init_params = TD.Parameters(theta, w)
        self.z = vect.zeros_like(self.init_params)

        self._define_forward(extractor)
        self._define_update()
        self.t = 0

    def _define_forward(self, extractor):
        # Make optimizer
        self.opt_init, self.opt_update, self.get_params = self.opt_cls(self.lr)
        self.opt_state = self.opt_init(self.init_params)

        def value_function(params, obs):
            x = extractor.forward(params.theta, obs)
            return jnp.squeeze(x.dot(params.w))
        self.value_function = value_function
        self.jit_value_function = jax.jit(value_function)

    def _define_update(self):
        discount = self.discount
        lambd = self.lambd

        @partial(jax.jit, static_argnames=['terminated', 'truncated'])
        def update(opt_state, target_params, z, obs, next_obs, reward, terminated, truncated, t):
            params = self.get_params(opt_state)
            v, grads = jax.value_and_grad(self.value_function)(params, obs)

            td_error = reward - v
            if not terminated:
                td_error += discount * self.value_function(params, next_obs)

            z = vect.etrace(z, discount * lambd, grads)
            step = vect.scale(-1 * td_error, z)
            opt_state = self.opt_update(t, step, opt_state)

            if terminated or truncated:
                z = vect.zeros_like(z)

            return opt_state, z

        self.update = update

    @property
    def params(self):
        return self.get_params(self.opt_state)

    def update_target_network(self):
        if self.target_period == 1:
            self.target_params = self.params
            return

        if (self.t % self.target_period) == 1:
            self.target_params = vect.copy(self.params)

    def predict(self, obs):
        return self.jit_value_function(self.params, obs)

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, info={}):
        self.t += 1
        self.update_target_network()
        self.opt_state, self.z = self.update(self.opt_state, self.target_params, self.z, obs, next_obs, reward, terminated, truncated, self.t)
