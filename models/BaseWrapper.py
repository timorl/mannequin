
from . import BaseModel

class BaseWrapper(BaseModel):
    def __init__(self, inner, *,
            inp_shape=None,
            out_shape=None,
            n_params=None,
            load_params=None,
            param_gradient=None,
            step=None):

        if inp_shape is None: inp_shape = inner.inp_shape
        if out_shape is None: out_shape = inner.out_shape
        if n_params is None: n_params = inner.n_params
        if load_params is None: load_params = inner.load_params
        if param_gradient is None: param_gradient = inner.param_gradient
        if step is None: step = inner.step

        self.get_input_shape = lambda: inp_shape
        self.get_output_shape = lambda: out_shape
        self.get_n_params = lambda: n_params
        self.load_params = load_params
        self.param_gradient = param_gradient
        self.step = step
