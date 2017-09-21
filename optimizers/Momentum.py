
from . import BaseOptimizer

class Momentum(BaseOptimizer):
    def __init__(self, value, *, lr, memory=0.9, print_norm=False):
        import numpy as np
        import os

        value = np.asarray(value)
        value.setflags(write=False)

        lr = float(lr)
        assert lr > 0.0

        running_mean = 0.0

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def apply_gradient(grad):
            nonlocal value, running_mean

            grad = np.asarray(grad)
            assert grad.shape == value.shape

            running_mean = running_mean * memory + grad * (1.0 - memory)
            update = lr * running_mean

            if print_norm:
                print("Update norm: %10.4f" % norm(update))

            value = value + update
            value.setflags(write=False)

        self.get_value = lambda: value
        self.apply_gradient = apply_gradient
