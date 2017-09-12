
from .BaseWorld import BaseWorld

class StochasticPolicy(BaseWorld):
    def __init__(self, inner):
        self.trajectories = lambda agent, n: inner.trajectories(
            StochasticPolicyAgent(agent),
            n
        )
        self.render = lambda a: inner.render(StochasticPolicyAgent(a))

from models.BaseWrapper import BaseWrapper

class StochasticPolicyAgent(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        rng = np.random.RandomState()

        def choice(v):
            assert len(v.shape) == 1
            i = rng.choice(len(v), p=v)
            v[:] = 0.0
            v[i] = 1.0
            return v

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)

            for i in range(len(outputs)):
                outputs[i] = choice(outputs[i])

            return states, outputs

        super().__init__(inner, step=step)
