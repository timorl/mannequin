
from . import BaseTwoMoments

class Adams(BaseTwoMoments):
    def __init__(self, value, *, lr,
            epsilon=1e-8,
            power=2.0,
            **params):
        import numpy as np

        assert (power >= 1.0) and (power <= 2.0)

        def update_rule(mean, var):
            return lr * (mean / (epsilon + np.power(var, power * 0.5)))

        super().__init__(value, update_rule, **params)
