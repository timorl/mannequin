
from . import verify_shapes
from .BaseWrapper import BaseWrapper

@verify_shapes
class Softmax(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        def softmax(v):
            v = np.exp(v - np.amax(v))
            return v / v.sum()

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)

            for i in range(len(outputs)):
                outputs[i] = softmax(outputs[i])

            return states, outputs

        super().__init__(inner, step=step)
