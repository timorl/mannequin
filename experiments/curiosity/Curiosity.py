
from worlds.BaseWorld import BaseWorld

class Curiosity(BaseWorld):
    def __init__(self, inner, classifier):
        import numpy as np

        def trajectories(agent, n):
            assert n >= 1
            assert agent==None
            trajs = inner.trajectories(None, n)
            _, pred = classifier.step([None]*len(trajs), [o for ((o, _, _),) in trajs])
            return [[([0,0],o,p[0])]for p, ((o, a, _),) in zip(pred, trajs)]

        self.trajectories = trajectories
