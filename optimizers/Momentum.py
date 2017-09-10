
from . import Optimizer

class RunningMean(object):
    def __init__(self, decay):
        decay = float(decay)
        assert decay > 0.0
        assert decay < 1.0

        biased_mean = 0.0
        decay_power = 1.0

        def update(value):
            nonlocal biased_mean, decay_power

            biased_mean = biased_mean * decay + value * (1.0 - decay)
            decay_power *= decay

        self.get = lambda: biased_mean / (1.0 - decay_power)
        self.update = update

class Momentum(Optimizer):
    def __init__(self, value, lr, decay, print_info=False):
        import numpy as np
        import os

        value = np.asarray(value)
        value.setflags(write=False)

        lr = float(lr)
        assert lr > 0.0

        running_mean = RunningMean(decay)

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def apply_gradient(grad):
            nonlocal value

            grad = np.asarray(grad)
            assert grad.shape == value.shape

            running_mean.update(grad)
            update = lr * running_mean.get()

            if print_info:
                print("Update norm: %10.4f" % norm(update))

            value = value + update
            value.setflags(write=False)

        self.get_value = lambda: value
        self.apply_gradient = apply_gradient
