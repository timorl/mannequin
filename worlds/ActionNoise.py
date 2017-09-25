
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

        def outputs(inputs):
            outputs = np.array(inner.outputs(inputs))

            n = inner.n_outputs
            outputs[:,:n] += rng.randn(len(outputs), n) * stddev
            return outputs

        super().__init__(inner, outputs=outputs)
