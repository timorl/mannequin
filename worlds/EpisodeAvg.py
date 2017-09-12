
from . import World

class EpisodeAvg(World):
    def __init__(self, inner):
        import numpy as np

        def process(traj):
            all_rew = [float(r) for o, a, r in traj]
            avg_rew = np.mean(all_rew)
            return [(o, a, avg_rew) for o, a, r in traj]

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)
            return [process(t) for t in trajs]

        self.trajectories = trajectories
        self.render = inner.render
