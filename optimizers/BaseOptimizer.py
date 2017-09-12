
class BaseOptimizer(object):
    # def __init__(self, value)
    #     ...

    def get_value(self): # -> {ndarray}
        raise NotImplementedError

    def apply_gradient(self, gradient):
        raise NotImplementedError
