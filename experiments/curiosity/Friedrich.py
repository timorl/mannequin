
from worlds.BaseWorld import BaseWorld

class Friedrich(BaseWorld):
    def __init__(self, gaussCenterer):
        import numpy as np

        def make_observation():
            if np.random.rand() < 0.5:
                #center = [2,0]
                _, (center,) = gaussCenterer.step([None],[[0,0]])
                ans = [1,0]
            else:
                center = [0,0]
                ans = [0,1]
            obs = center + np.random.randn(2)*0.2
            obs += 1001.
            obs = np.abs(obs)
            obs %= 2.
            obs -= 1.
            return obs, ans, 1.

        def trajectories(agent, n):
            assert n >= 1
            assert agent==None
            return [[make_observation()] for _ in range(n)]

        self.trajectories = trajectories
