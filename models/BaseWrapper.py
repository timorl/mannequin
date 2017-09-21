
from . import BaseModel

class BaseWrapper(BaseModel):
    def __init__(self, inner, **kwargs):
        self.get_input_shape = inner.get_input_shape
        self.get_output_shape = inner.get_output_shape
        self.get_n_params = inner.get_n_params
        self.load_params = inner.load_params
        self.param_gradient = inner.param_gradient
        self.step = inner.step

        for name, value in kwargs.items():
            assert hasattr(self, name)
            setattr(self, name, value)
