
from .retrace import retrace

def policy_gradient(trajs, *, policy):
    import numpy as np

    # TODO: backprop through states!

    learn_inps = []
    learn_grads = []

    policy_trajs = retrace(trajs, model=policy, with_inputs=True)

    for traj, policy_traj in zip(trajs, policy_trajs):
        # Unpack trajectories to vertical arrays
        _, real_outs, rews = zip(*traj)
        inps, policy_outs = zip(*policy_traj)

        # Verify array shapes
        real_outs = np.asarray(real_outs)
        policy_outs = np.asarray(policy_outs)
        assert policy_outs.shape == real_outs.shape
        rews = np.asarray(rews)
        assert rews.shape == (len(rews),)

        # True off-policy version of REINFORCE (!!!)
        grads = np.multiply((real_outs - policy_outs).T, rews.T).T
        if policy.n_states >= 1:
            # There is no gradient for the most recent state
            grads = np.concatenate(
                (grads, np.zeros((len(rews), policy.n_states))),
                axis=1
            )

        learn_inps += inps
        learn_grads += list(grads)

    return policy.param_gradient(learn_inps, learn_grads)
