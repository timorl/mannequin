
from . import Model

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
