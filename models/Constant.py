
from . import BaseModel
from ._verify_shapes import verify_shapes

@verify_shapes
class Constant(BaseModel):
    def __init__(self, size):
        import numpy as np

        size = int(size)
        self.get_n_inputs = lambda: 0
        self.get_n_outputs = lambda: size
        self.get_n_states = lambda: 0
        self.get_n_params = lambda: size

        value = None

        def load_params(params):
            nonlocal value
            value = np.array(params).reshape((size,))

        def outputs(inputs):
            if value is None:
                raise ValueError("Call load_params() first")
            return np.array([value] * len(inputs))

        def param_gradient(inputs, output_gradients):
            output_gradients = np.array(output_gradients)
            assert output_gradients.shape == (len(inputs), size)
            return np.mean(output_gradients, axis=0)

        self.load_params = load_params
        self.outputs = outputs
        self.param_gradient = param_gradient
