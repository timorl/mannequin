
from .BaseWorld import BaseWorld
from execute import retrace

class Accuracy(BaseWorld):
    def __init__(self, inner):
        import numpy as np

        def process(experience, agent_act):
            obs, act, rew = experience
            act = np.asarray(act)
            assert act.shape == agent_act.shape

            # Make sure it's a supervised dataset
            assert (rew > 0.99) and (rew < 1.01)

            if np.argmax(act) == np.argmax(agent_act):
                return obs, agent_act, 100.0
            else:
                return obs, agent_act, 0.0

        def trajectories(agent, n):
            trajs = inner.trajectories(None, n)
            return [
                [process(exp, o) for exp, (s, o) in zip(traj, a_traj)]
                for traj, a_traj in zip(trajs, retrace(agent, trajs))
            ]

        self.trajectories = trajectories
        self.render = inner.render
