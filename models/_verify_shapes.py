
from . import Model

def verify_shapes(model_cls):
    import numpy as np

    def init(self, *args, **kwargs):
        inner = model_cls(*args, **kwargs)

        inp_shape = tuple(max(1, int(d)) for d in inner.inp_shape)
        out_shape = tuple(max(1, int(d)) for d in inner.out_shape)
        n_params = int(inner.get_n_params())

        self.get_input_shape = lambda: inp_shape
        self.get_output_shape = lambda: out_shape
        self.get_n_params = lambda: n_params

        def load_params(params):
            params = np.asarray(params)
            assert params.shape == (n_params,)
            inner.load_params(params)

        def param_gradient(states, inputs, gradients):
            batch = len(states)
            assert batch >= 1

            inputs = np.asarray(inputs)
            assert inputs.shape == (batch,) + inp_shape

            gradients = np.asarray(gradients)
            assert gradients.shape == (batch,) + out_shape

            grad = inner.param_gradient(states, inputs, gradients)
            grad = np.asarray(grad)
            assert grad.shape == (n_params,)

            return grad

        def step(states, inputs):
            batch = len(states)
            assert batch >= 1

            inputs = np.asarray(inputs)
            assert inputs.shape == (batch,) + inp_shape

            states, outputs = inner.step(states, inputs)
            outputs = np.asarray(outputs)
            assert outputs.shape == (batch,) + out_shape
            assert len(states) == batch

            return states, outputs

        self.load_params = load_params
        self.param_gradient = param_gradient
        self.step = step

    class Verified(Model):
        __init__ = init
        __name__ = model_cls.__name__
        __qualname__ = model_cls.__qualname__

    return Verified
