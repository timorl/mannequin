
class Optimizer(object):
    # def __init__(self, value)
    #     ...

    def get_info(self): # -> {str}
        raise NotImplementedError

    def get_best_value(self): # -> {ndarray}
        raise NotImplementedError

    def get_requests(self): # -> {list of ndarray}
        raise NotImplementedError

    def feed_gradients(self, gradients):
        raise NotImplementedError

from .Adam import Adam
from .Momentum import Momentum
