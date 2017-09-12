
from . import Model
from ._verify_shapes import verify_shapes

@verify_shapes
class Constant(Model):
    def __init__(self, inp_shape, out_shape):
        import numpy as np

        inp_shape = tuple(inp_shape)
        out_shape = tuple(out_shape)
        self.get_input_shape = lambda: inp_shape
        self.get_output_shape = lambda: out_shape
        self.get_n_params = lambda: np.prod(out_shape)

        value = None

        def load_params(params):
            nonlocal value
            value = np.array(params).reshape(out_shape)

        def param_gradient(states, inputs, output_gradients):
            output_gradients = np.array(output_gradients)
            assert output_gradients.shape == (len(states),) + out_shape
            return np.mean(output_gradients, axis=0).reshape(-1)

        def step(states, inputs):
            if value is None:
                raise ValueError("Call load_params() first")
            return states, np.array([value] * len(states))

        self.load_params = load_params
        self.param_gradient = param_gradient
        self.step = step
