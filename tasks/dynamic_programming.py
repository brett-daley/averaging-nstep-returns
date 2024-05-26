import numpy as np


def policy_evaluation(env, discount, policy, precision=1e-3):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    S = env.observation_space.n
    A = env.action_space.n
    Q = np.zeros([S, A], dtype=np.float64)

    def bellman(Q):
        for s in env.states():
            for a in env.actions():
                next_states, rewards, dones, probs = env.model(s, a)

                Vp = np.array([np.dot(policy(sp), Q[sp]) for sp in next_states])
                Vp *= np.where(dones, 0.0, discount)

                Q[s,a] = np.sum(probs * (rewards + Vp))

    while True:
        Q_old = Q.copy()
        bellman(Q)

        if np.abs(Q - Q_old).max() <= precision:
            return np.array([np.dot(policy(s), Q[s]) for s in env.states()])
