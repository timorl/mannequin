
from . import BaseWorld
from models import BaseWrapper

class StochasticPolicy(BaseWorld):
    def __init__(self, inner):
        self.trajectories = lambda agent, n: inner.trajectories(
            StochasticPolicyAgent(agent),
            n
        )
        self.render = lambda a: inner.render(StochasticPolicyAgent(a))

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
