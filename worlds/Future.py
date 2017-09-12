
from . import World

class Future(World):
    def __init__(self, inner, *, horizon):
        import numpy as np

        def process(traj):
            rew_sum = 0.0
            reversed_out = []

            for o, a, r in reversed(traj):
                rew_sum = rew_sum * (1.0 - (1.0/horizon)) + float(r)
                reversed_out.append((o, a, rew_sum))

            return list(reversed(reversed_out))

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)
            return [process(t) for t in trajs]

        self.trajectories = trajectories
