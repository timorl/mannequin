
class Model:
    def get_input_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_output_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_reward_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_n_params(self): # -> {int}
        raise NotImplementedError

    def param_load(self, params):
        raise NotImplementedError

    def param_gradient(self, trajectories): # -> {ndarray}
        raise NotImplementedError

    def step(states, inputs): # -> (states, outputs)
        raise NotImplementedError

    # For convenience only
    def __getattr__(self, name):
        if name in ("inp_shape", "input_shape"):
            return self.get_input_shape()
        if name in ("out_shape", "output_shape"):
            return self.get_output_shape()
        if name in ("rew_shape", "reward_shape"):
            return self.get_reward_shape()
        if name in ("n_params"):
            return self.get_n_params()
        return object.__getattribute__(self, name)

from .BasicNet import BasicNet
from .Softmax import Softmax
