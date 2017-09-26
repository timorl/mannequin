
from . import BaseModel

class BaseWrapper(BaseModel):
    def __init__(self, inner, *,
            n_inputs=None,
            n_outputs=None,
            n_states=None,
            n_params=None,
            load_params=None,
            outputs=None,
            param_gradient=None,
            input_gradients=None):

        if n_inputs is None: n_inputs = inner.n_inputs
        if n_outputs is None: n_outputs = inner.n_outputs
        if n_states is None: n_states = inner.n_states
        if n_params is None: n_params = inner.n_params
        if load_params is None: load_params = inner.load_params
        if outputs is None: outputs = inner.outputs
        if param_gradient is None: param_gradient = inner.param_gradient
        if input_gradients is None: input_gradients = inner.input_gradients

        self.get_n_inputs = lambda: n_inputs
        self.get_n_outputs = lambda: n_outputs
        self.get_n_states = lambda: n_states
        self.get_n_params = lambda: n_params
        self.load_params = load_params
        self.outputs = outputs
        self.param_gradient = param_gradient
        self.input_gradients = input_gradients
