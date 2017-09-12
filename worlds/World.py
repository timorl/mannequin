
class World(object):
    # Note: worlds are assumed to be thread-safe
    # (calls to trajectories() can be concurrent)

    def trajectories(self, agent, n):
        # Returns a list of n trajectories, where a trajectory
        # is a list of tuples (observation, action, reward),
        # and each tuple has types (ndarray, ndarray, float)
        raise NotImplementedError
