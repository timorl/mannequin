
from . import World

class Future(World):
    def __init__(self, inner, horizon):
        import numpy as np

        def process(traj):
            rew_sum = 0.0

            for o, a, r in reversed(traj):
                assert len(r) == 1
                assert isinstance(r[0], (float, np.float32))
                rew_sum = rew_sum * (1.0 - (1.0 / horizon)) + r[0]
                r[0] = rew_sum

            return traj

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)
            return [process(t) for t in trajs]

        self.trajectories = trajectories
