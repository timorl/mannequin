
from . import Optimizer

class Momentum(Optimizer):
    def __init__(self, value, lr, momentum):
        import numpy as np
        import os

        value = np.asarray(value)
        value.setflags(write=False)

        lr = float(lr)
        assert lr > 0.0

        momentum = float(momentum)
        assert momentum > 0.0
        assert momentum < 1.0

        update = 0.0

        def norm(v):
            return np.sqrt(np.sum(np.square(v)))

        def get_info():
            return ("last update: %.2f" % norm(update))

        def get_requests():
            return [value]

        def feed_gradients(gradients):
            nonlocal value, update

            assert len(gradients) == 1
            grad = np.asarray(gradients[0])
            assert grad.shape == value.shape

            update *= momentum
            update += grad * (lr * (1.0 - momentum))

            value = value + update
            value.setflags(write=False)

            return "update norm="

        self.get_info = get_info
        self.get_best_value = lambda: value
        self.get_requests = get_requests
        self.feed_gradients = feed_gradients
