
from . import BaseModel

def verify_shapes(model_cls):
    import numpy as np

    def init(self, *args, **kwargs):
        inner = model_cls(*args, **kwargs)

        self.get_n_inputs = lambda: inner.n_inputs
        self.get_n_outputs = lambda: inner.n_outputs
        self.get_n_states = lambda: inner.n_states
        self.get_n_params = lambda: inner.n_params

        def verify(a, shape):
            if shape == None:
                return None
            a = np.asarray(a)
            if a.shape != shape:
                raise ValueError("Invalid shape: %s, expected: %s"
                    % (str(a.shape), str(shape)))
            return a

        def params(a):
            return verify(a, (inner.n_params,))

        def inputs(batch, a):
            if inner.n_inputs <= 0 and inner.n_states <= 0:
                return [None] * batch
            return verify(a, (batch, inner.n_inputs + inner.n_states))

        def outputs(batch, a):
            return verify(a, (batch, inner.n_outputs + inner.n_states))

        self.load_params = lambda p: (
            (inner.load_params(params(p)), None)[1]
        )

        self.outputs = lambda i: (
            outputs(len(i), inner.outputs(inputs(len(i), i)))
        )

        self.param_gradient = lambda i, o: (
            params(inner.param_gradient(
                inputs(len(i), i),
                outputs(len(i), o)
            ))
        )

        self.input_gradients = lambda i, o: (
            inputs(len(i), inner.input_gradients(
                inputs(len(i), i),
                outputs(len(i), o)
            ))
        )

        for name in inner.__dict__:
            if name.startswith("_") and not name.startswith("__"):
                setattr(self, name, inner.__dict__[name])

    class Verified(BaseModel):
        __init__ = init
        __name__ = model_cls.__name__
        __qualname__ = model_cls.__qualname__

    return Verified
