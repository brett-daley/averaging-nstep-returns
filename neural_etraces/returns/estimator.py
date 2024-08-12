
class Estimator:
    def traj_len(self, batch_len):
        return 2 * batch_len

    def calc_returns(self, *args, **kwargs):
        raise NotImplementedError
