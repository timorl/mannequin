
from . import World

class Normalized(World):
    def __init__(self, inner):
        import sys
        import numpy as np

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)

            all_rewards = []
            for t in trajs:
                for o, a, r in t:
                    assert len(r) == 1
                    assert isinstance(r[0], (float, np.float32))
                    all_rewards.append(r[0])

            assert len(all_rewards) >= 2
            avg = np.mean(all_rewards)
            std = np.std(all_rewards)
            if std < 0.000001:
                std = 1.0
                sys.stderr.write("Normalize: all rewards are equal\n")

            for t in trajs:
                for o, a, r in t:
                    r[0] = (r[0] - avg) / std

            return trajs

        self.trajectories = trajectories
