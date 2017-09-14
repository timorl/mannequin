
from .BaseWorld import BaseWorld
from trajectories import accuracy

class Accuracy(BaseWorld):
    def __init__(self, inner):
        def trajectories(agent, n):
            return accuracy(inner.trajectories(None, n), model=agent)

        self.trajectories = trajectories
        self.render = inner.render
