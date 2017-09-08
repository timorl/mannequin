
from .BaseWrapper import BaseWrapper

class Softmax(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        super().__init__(inner)

        def softmax(v):
            v = np.exp(v - np.amax(v))
            return v / v.sum()

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)

            for i in range(len(outputs)):
                outputs[i] = softmax(outputs[i])

            return states, outputs

        self.step = step
