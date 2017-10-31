
from . import BaseOptimizer

class BaseTwoMoments(BaseOptimizer):
    def __init__(self, value, update_rule, *,
            memory=0.9,
            var_memory=0.999,
            print_norm=False):
        import numpy as np
        import os

        value = np.asarray(value, dtype=np.float32)
        value.setflags(write=False)

        running_mean = RunningMean(memory)
        running_var = RunningMean(var_memory)

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def apply_gradient(grad):
            nonlocal value

            grad = np.asarray(grad)
            assert grad.shape == value.shape

            running_mean.update(grad)
            running_var.update(np.square(grad))

            add = update_rule(running_mean.get(), running_var.get())

            if print_norm:
                print("Update norm: %10.4f" % norm(add))

            value = value + add
            value.setflags(write=False)

        self.get_value = lambda: value
        self.apply_gradient = apply_gradient

class RunningMean(object):
    def __init__(self, memory):
        memory = float(memory)
        assert memory >= 0.0
        assert memory < 1.0

        biased_mean = 0.0
        memory_power = 1.0

        def update(value):
            nonlocal biased_mean, memory_power

            biased_mean = biased_mean * memory + value * (1.0 - memory)
            memory_power *= memory

        self.get = lambda: biased_mean / (1.0 - memory_power)
        self.update = update
