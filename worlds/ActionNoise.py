
from . import BaseWorld
from models import BaseWrapper

class ActionNoise(BaseWorld):
    def __init__(self, inner, *, stddev):
        def wrap(agent):
            if agent is None:
                return None
            else:
                return ActionNoiseAgent(agent, stddev=stddev)

        self.trajectories = lambda a, n: inner.trajectories(wrap(a), n)
        self.render = lambda a: inner.render(wrap(a))

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
