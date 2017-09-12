
from . import retrace

def policy_gradient(model, trajs):
    import numpy as np

    learn_states = []
    learn_inps = []
    learn_grads = []

    for traj, model_traj in zip(trajs, retrace(model, trajs)):
        # Unpack trajectories to vertical arrays
        inps, real_outs, rews = zip(*traj)
        states, model_outs = zip(*model_traj)

        # Verify array shapes
        real_outs = np.asarray(real_outs)
        model_outs = np.asarray(model_outs)
        assert model_outs.shape == real_outs.shape
        rews = np.asarray(rews)
        assert rews.shape == (len(rews),)

        # True off-policy version of REINFORCE (!!!)
        grads = np.multiply((real_outs - model_outs).T, rews.T).T

        learn_states += states
        learn_inps += inps
        learn_grads += list(grads)

    return model.param_gradient(learn_states, learn_inps, learn_grads)
