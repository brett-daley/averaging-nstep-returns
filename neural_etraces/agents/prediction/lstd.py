from functools import partial

import jax
import jax.numpy as jnp

from agents.prediction.gradient_lstd import GradientLSTD
import extractors


class LSTD(GradientLSTD):
    def __init__(self, observation_space, action_space, seed, discount, **kwargs):
        super().__init__(observation_space, action_space, seed, discount, **kwargs)
        assert isinstance(self.extractor, extractors.Identity)
        self.w = self.params.w

    def _define_update(self, extractor):
        super()._define_update(extractor)
        discount = self.discount

        @partial(jax.jit, static_argnames=['terminated', 'truncated'])
        def update(A, b, obs, next_obs, reward, terminated, truncated, t):
            x = obs

            # Compute negative gradient of TD error
            neg_grad_td_error = x
            if not terminated:
                xp = next_obs
                neg_grad_td_error -= discount * xp

            # Update stats
            A += jnp.outer(x, neg_grad_td_error)
            b += reward * x

            w, _, _, _ = jnp.linalg.lstsq(A, b)
            return w, A, b

        self.update = update

    def predict(self, obs):
        return jnp.dot(obs, self.w)

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, info={}):
        self.t += 1
        obs, next_obs = map(jnp.array, [obs, next_obs])
        self.w, self.A, self.b = self.update(self.A, self.b, obs, next_obs, reward, terminated, truncated, self.t)
