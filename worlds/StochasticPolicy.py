
from . import BaseWorld
from models import BaseWrapper

class StochasticPolicy(BaseWorld):
    def __init__(self, inner):
        def wrap(agent):
            if agent is None:
                return None
            else:
                return StochasticPolicyAgent(agent)

        self.trajectories = lambda a, n: inner.trajectories(wrap(a), n)
        self.render = lambda a: inner.render(wrap(a))

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

        def outputs(inputs):
            outputs = np.array(inner.outputs(inputs))

            n = inner.n_outputs
            for i in range(len(outputs)):
                outputs[i,:n] = choice(outputs[i,:n])

            return outputs

        super().__init__(inner, outputs=outputs)
