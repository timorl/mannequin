
from .BaseWorld import BaseWorld

class ActionNoise(BaseWorld):
    def __init__(self, inner, *, stddev):
        self.trajectories = lambda agent, n: inner.trajectories(
            ActionNoiseAgent(agent, stddev),
            n
        )
        self.render = lambda a: inner.render(ActionNoiseAgent(a))

from models.BaseWrapper import BaseWrapper

class ActionNoiseAgent(BaseWrapper):
    def __init__(self, inner, stddev):
        import numpy as np

        rng = np.random.RandomState()

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)
            outputs += rng.randn(*outputs.shape) * stddev
            return states, outputs

        super().__init__(inner, step=step)
