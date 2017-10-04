
from . import BaseWorld

class BaseTable(BaseWorld):
    def __init__(self, x, y):
        import numpy as np

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        assert len(x) == len(y)
        assert len(x) >= 1

        x.setflags(write=False)
        y.setflags(write=False)

        rng = np.random.RandomState()

        def trajectories(agent, n):
            assert agent == None
            assert n >= 1

            return [
                [(x[i], y[i], 1.0)]
                for i in rng.choice(len(x), size=n)
            ]

        self.trajectories = trajectories
