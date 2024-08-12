from functools import partial

import jax
import jax.numpy as jnp

from agents.prediction.td import TD


class GradientLSTD(TD):
    def __init__(self, observation_space, action_space, seed, discount, **kwargs):
        super().__init__(observation_space, action_space, seed, discount, **kwargs)
        if self.lambd != 0.0:
            raise NotImplementedError
        features = self.init_params.w.shape[0]
        self.A = jnp.zeros([features, features])
        self.b = jnp.zeros(features)

    def _define_update(self, extractor):
        super()._define_update(extractor)
        get_params = self.get_params
        discount = self.discount

        @partial(jax.jit, static_argnames=['terminated', 'truncated'])
        def update(opt_state, A, b, obs, next_obs, reward, terminated, truncated, t):
            params = get_params(opt_state)
            x = extractor.forward(params.theta, obs)

            # Compute negative gradient of TD error
            neg_grad_td_error = x
            if not terminated:
                xp = extractor.forward(params.theta, next_obs)
                neg_grad_td_error -= discount * xp

            # Update stats
            A += jnp.outer(x, neg_grad_td_error)
            b += reward * x

            def loss(params):
                w = params.w
                # Normalize stats for optimization
                A_mean = jax.lax.stop_gradient(A / t)
                b_mean = jax.lax.stop_gradient(b / t)
                return jnp.mean(0.5 * jnp.square(A_mean @ w - b_mean))

            grads = jax.grad(loss)(params)
            opt_state = self.opt_update(t, grads, opt_state)

            return opt_state, A, b

        self.update = update

    def reinforce(self, obs, action, next_obs, reward, terminated, truncated, info={}):
        self.t += 1
        obs, next_obs = map(jnp.array, [obs, next_obs])
        self.opt_state, self.A, self.b = self.update(self.opt_state, self.A, self.b, obs, next_obs, reward, terminated, truncated, self.t)
