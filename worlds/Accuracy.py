
from . import World

class Accuracy(World):
    def __init__(self, inner):
        import numpy as np

        def process(o, a, r):
            assert a.shape == r.shape
            answer = np.argmax(a)
            wanted = np.argmax(a + r)
            return o, a, [100.0] if answer == wanted else [0.0]

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)
            return [[process(*e) for e in t] for t in trajs]

        self.trajectories = trajectories
