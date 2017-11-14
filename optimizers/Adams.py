
from . import BaseTwoMoments

class Adams(BaseTwoMoments):
    def __init__(self, value, *, lr,
            memory=0.9,
            var_memory=0.98,
            epsilon=1e-8,
            **params):
        import numpy as np

        def update_rule(mean, var):
            return lr * (mean / (epsilon + var))

        super().__init__(value, update_rule, memory=memory,
            var_memory=var_memory, **params)
