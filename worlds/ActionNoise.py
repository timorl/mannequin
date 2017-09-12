
from . import World

class ActionNoise(World):
    def __init__(self, inner, *, stddev):
        self.trajectories = lambda agent, n: inner.trajectories(
            NoisyAgent(agent, stddev),
            n
        )

from models.BaseWrapper import BaseWrapper

class NoisyAgent(BaseWrapper):
    def __init__(self, inner, stddev):
        import numpy as np

        rng = np.random.RandomState()

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)
            outputs += rng.randn(*outputs.shape) * stddev
            return states, outputs

        super().__init__(inner, step=step)
