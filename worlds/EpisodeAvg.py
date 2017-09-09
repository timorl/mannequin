
from . import World

class EpisodeAvg(World):
    def __init__(self, inner):
        import numpy as np

        def process(traj):
            rew_sum = 0.0
            for o, a, r in traj:
                assert len(r) == 1
                assert isinstance(r[0], (float, np.float32))
                rew_sum += r[0]

            rew_sum /= len(traj)
            for o, a, r in traj:
                r[0] = rew_sum

            return traj

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)
            return [process(t) for t in trajs]

        self.trajectories = trajectories
