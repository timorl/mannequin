
from . import BaseWrapper

class Softmax(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        def softmax(v):
            v = np.exp(v - np.amax(v))
            return v / v.sum()

        def outputs(inputs):
            outputs = np.array(inner.outputs(inputs))

            n = inner.n_outputs
            for i in range(len(outputs)):
                outputs[i,:n] = softmax(outputs[i,:n])

            return outputs

        super().__init__(inner, outputs=outputs)
