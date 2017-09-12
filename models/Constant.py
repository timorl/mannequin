
from .BaseModel import BaseModel
from ._verify_shapes import verify_shapes

@verify_shapes
class Constant(BaseModel):
    def __init__(self, *shape):
        import numpy as np

        shape = tuple(max(1, int(d)) for d in shape)
        self.get_input_shape = lambda: None
        self.get_output_shape = lambda: shape
        self.get_n_params = lambda: np.prod(shape)

        value = None

        def load_params(params):
            nonlocal value
            value = np.array(params).reshape(shape)

        def param_gradient(states, inputs, output_gradients):
            output_gradients = np.array(output_gradients)
            assert output_gradients.shape == (len(states),) + shape
            return np.mean(output_gradients, axis=0).reshape(-1)

        def step(states, inputs):
            if value is None:
                raise ValueError("Call load_params() first")
            return states, np.array([value] * len(states))

        self.load_params = load_params
        self.param_gradient = param_gradient
        self.step = step
