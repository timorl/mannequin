
from .retrace import retrace

def accuracy(trajs, *, model, percent=False):
    import numpy as np

    def process(experience, agent_act):
        obs, act, rew = experience
        act = np.asarray(act)
        assert act.shape == agent_act.shape

        # Make sure it's a supervised dataset
        assert (rew > 0.99) and (rew < 1.01)

        if np.argmax(act) == np.argmax(agent_act):
            return obs, agent_act, (100.0 if percent else 1.0)
        else:
            return obs, agent_act, 0.0

    return [
        [process(exp, o) for exp, o in zip(traj, outs)]
        for traj, outs in zip(trajs, retrace(trajs, model=model))
    ]
