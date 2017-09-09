
class Model:
    def get_input_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_output_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_reward_shape(self): # -> {tuple of int}
        raise NotImplementedError

    def get_n_params(self): # -> {int}
        raise NotImplementedError

    def load_params(self, params):
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

def verify_shapes(model_cls):
    import numpy as np

    def init(self, *args, **kwargs):
        inner = model_cls(*args, **kwargs)

        inp_shape = tuple(int(d) for d in inner.get_input_shape())
        out_shape = tuple(int(d) for d in inner.get_output_shape())
        rew_shape = tuple(int(d) for d in inner.get_reward_shape())
        n_params = int(inner.get_n_params())

        self.get_input_shape = lambda: inp_shape
        self.get_output_shape = lambda: out_shape
        self.get_reward_shape = lambda: rew_shape
        self.get_n_params = lambda: n_params

        def load_params(params):
            params = np.asarray(params)
            assert params.shape == (n_params,)
            inner.load_params(params)

        def param_gradient(trajectories):
            for t in trajectories:
                assert isinstance(t, list)
                assert len(t) >= 1
                for i in range(len(t)):
                    assert isinstance(t[i], tuple)
                    assert len(t[i]) == 3
                    if (not isinstance(t[i][0], np.ndarray)
                        or not isinstance(t[i][1], np.ndarray)
                        or not isinstance(t[i][2], np.ndarray)):
                        t[i] = tuple(map(np.asarray, t[i]))
                    assert t[i][0].shape == inp_shape
                    assert t[i][1].shape == out_shape
                    assert t[i][2].shape == rew_shape
            grad = inner.param_gradient(trajectories)
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

from .BasicNet import BasicNet
from .OffPolicy import OffPolicy
from .RandomChoice import RandomChoice
from .Softmax import Softmax
