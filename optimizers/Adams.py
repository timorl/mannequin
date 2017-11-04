
from . import BaseTwoMoments

class Adams(BaseTwoMoments):
    def __init__(self, value, *, lr,
            epsilon=1e-8,
            **params):
        import numpy as np

        def update_rule(mean, var):
            return lr * (mean / (epsilon + var))

        super().__init__(value, update_rule, **params)
