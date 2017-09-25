
from .retrace import retrace

def policy_gradient(trajs, *, policy):
    import numpy as np

    # TODO: support states!

    learn_inps = []
    learn_grads = []

    policy_trajs = retrace(trajs, model=policy)

    for traj, policy_outs in zip(trajs, policy_trajs):
        # Unpack trajectories to vertical arrays
        inps, real_outs, rews = zip(*traj)

        # Verify array shapes
        real_outs = np.asarray(real_outs)
        policy_outs = np.asarray(policy_outs)
        assert policy_outs.shape == real_outs.shape
        rews = np.asarray(rews)
        assert rews.shape == (len(rews),)

        # True off-policy version of REINFORCE (!!!)
        grads = np.multiply((real_outs - policy_outs).T, rews.T).T

        learn_inps += inps
        learn_grads += list(grads)

    return policy.param_gradient(learn_inps, learn_grads)
