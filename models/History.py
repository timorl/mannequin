
from . import BaseWrapper

class History(BaseWrapper):
    def __init__(self, inner, *, length):
        import numpy as np

        assert length >= 2
        n_inputs = int(inner.n_inputs)
        assert n_inputs % length == 0
        n_inputs = n_inputs // length

        def hist(inputs):
            inputs = np.asarray(inputs)
            assert len(inputs.shape) == 2
            assert inputs.shape[1] % n_inputs == 0
            if inputs.shape[1] < inner.n_inputs:
                inputs = np.pad(
                    inputs,
                    ((0, 0), (0, inner.n_inputs - inputs.shape[1])),
                    "constant"
                )
            assert inputs.shape[1] == inner.n_inputs
            return inputs

        def outputs(inputs):
            inputs = hist(inputs)
            outputs = np.asarray(inner.outputs(inputs))
            assert outputs.shape[1] == inner.n_outputs

            # Save the last length-1 input batches as state
            inputs = inputs[:,:((length-1) * n_inputs)]
            return np.concatenate((outputs, inputs), axis=1)

        def param_gradient(inputs, output_gradients):
            return inner.param_gradient(hist(inputs), output_gradients)

        def input_gradients(inputs, output_gradients):
            return inner.input_gradients(hist(inputs), output_gradients)

        super().__init__(inner,
            n_inputs=n_inputs,
            outputs=outputs,
            param_gradient=param_gradient,
            input_gradients=input_gradients)
