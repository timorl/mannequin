
from . import Model

class BaseWrapper(Model):
    def __init__(self, inner):
        self.get_input_shape = inner.get_input_shape
        self.get_output_shape = inner.get_output_shape
        self.get_reward_shape = inner.get_reward_shape
        self.get_n_params = inner.get_n_params
        self.param_load = inner.param_load
        self.param_gradient = inner.param_gradient
        self.step = inner.step
