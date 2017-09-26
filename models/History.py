
from . import BaseWrapper
from ._verify_shapes import verify_shapes

@verify_shapes
class History(BaseWrapper):
    def __init__(self, inner, *, length):
        import numpy as np

        assert length >= 2
        single_frame = int(inner.n_inputs) // length
        assert single_frame * length == inner.n_inputs

        def outputs(inputs):
            outputs = np.asarray(inner.outputs(inputs))

            # Make sure the inner model has no state
            assert outputs.shape[1] == inner.n_outputs

            # Save the last length-1 input batches as state
            inputs = inputs[:,:((length-1) * single_frame)]
            return np.concatenate((outputs, inputs), axis=1)

        def param_gradient(inputs, output_gradients):
            output_gradients = output_gradients[:,:inner.n_outputs]
            return inner.param_gradient(inputs, output_gradients)

        def input_gradients(inputs, output_gradients):
            output_gradients = output_gradients[:,:inner.n_outputs]
            return inner.input_gradients(inputs, output_gradients)

        super().__init__(inner,
            n_inputs=single_frame,
            n_states=(length-1) * single_frame,
            outputs=outputs,
            param_gradient=param_gradient,
            input_gradients=input_gradients)
