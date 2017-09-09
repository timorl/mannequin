
class World:
    def trajectories(self, agent, n):
        # Returns a list of n trajectories, where each trajectory
        # is a list of tuples: (observation, action, reward)
        raise NotImplementedError

from .Accuracy import Accuracy
from .EpisodeAvg import EpisodeAvg
from .Future import Future
from .Gym import Gym
from .Mnist import Mnist
from .Normalized import Normalized
