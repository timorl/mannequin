
from . import BaseWorld
from models import BaseWrapper

class ActionNoise(BaseWorld):
    def __init__(self, inner, *, stddev):
        self.trajectories = lambda agent, n: inner.trajectories(
            ActionNoiseAgent(agent, stddev=stddev),
            n
        )
        self.render = lambda agent: inner.render(
            ActionNoiseAgent(agent, stddev=stddev)
        )

class ActionNoiseAgent(BaseWrapper):
    def __init__(self, inner, *, stddev):
        import numpy as np

        rng = np.random.RandomState()

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)
            outputs += rng.randn(*outputs.shape) * stddev
            return states, outputs

        super().__init__(inner, step=step)
