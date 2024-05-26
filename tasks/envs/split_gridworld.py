from gym_classics.envs.abstract.gridworld import Gridworld


class SplitGridworld(Gridworld):
    def __init__(self):
        dims = (5, 3)
        W, H = dims
        assert W % 2 == 1
        assert H % 2 == 1
        mid_W = W // 2
        mid_H = H // 2
        start = (0, mid_H)         # Start at far left, centered vertically
        self._goal = (W-1, mid_H)  # Goal at far right, centered vertically
        blocks = {
            # Wall runs vertically through center of gridworld, with a single opening in the middle
            (mid_W, i) for i in range(H) if i != mid_H
        }
        super().__init__(n_actions=4, dims=dims, starts={start}, blocks=blocks)

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return state == self._goal
