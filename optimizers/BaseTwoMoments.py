
from . import Optimizer

class BaseTwoMoments(Optimizer):
    def __init__(self, value, update_rule, *,
            mean_decay=0.9,
            var_decay=0.999,
            print_info=False):
        import numpy as np
        import os

        value = np.asarray(value, dtype=np.float32)
        value.setflags(write=False)

        running_mean = RunningMean(mean_decay)
        running_var = RunningMean(var_decay)

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def apply_gradient(grad):
            nonlocal value

            grad = np.asarray(grad)
            assert grad.shape == value.shape

            running_mean.update(grad)
            running_var.update(np.square(grad))

            add = update_rule(running_mean.get(), running_var.get())

            if print_info:
                print("Update norm: %10.4f" % norm(add))

            value = value + add
            value.setflags(write=False)

        self.get_value = lambda: value
        self.apply_gradient = apply_gradient

class RunningMean(object):
    def __init__(self, decay):
        decay = float(decay)
        assert decay >= 0.0
        assert decay < 1.0

        biased_mean = 0.0
        decay_power = 1.0

        def update(value):
            nonlocal biased_mean, decay_power

            biased_mean = biased_mean * decay + value * (1.0 - decay)
            decay_power *= decay

        self.get = lambda: biased_mean / (1.0 - decay_power)
        self.update = update
