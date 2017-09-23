
from .retrace import retrace

def cross_entropy(trajs, *, model, negative=False):
    import numpy as np

    def process(experience, agent_act):
        obs, act, rew = experience
        act = np.asarray(act)
        assert act.shape == agent_act.shape

        # Make sure it's a probability distribution
        act_sum = np.sum(act)
        assert (act_sum > 0.99) and (act_sum < 1.01)

        result = -np.sum(
            np.multiply(act, np.log2(agent_act))
        )

        return obs, agent_act, -result if negative else result

    return [
        [process(exp, o) for exp, o in zip(traj, outs)]
        for traj, outs in zip(trajs, retrace(trajs, model=model))
    ]
