
from .retrace import retrace

def policy_gradient(trajs, *, policy, state_grad_clip=1.0):
    if policy.n_states <= 0:
        return policy_gradient_stateless(trajs, policy=policy)

    import numpy as np

    grad_sum = 0.0
    grad_count = 0

    policy_trajs = retrace(trajs, model=policy, with_inputs=True)

    # Final states have no gradients
    state_grads = [np.zeros(policy.n_states)] * len(trajs)

    # Start from the last step and move backwards in time
    step = np.max([len(t) for t in trajs])

    while step >= 1:
        step -= 1

        # Find trajectories that are active at this time
        pick = [p for p, t in enumerate(trajs) if len(t) > step]

        inps = [policy_trajs[p][step][0] for p in pick]
        rews = [trajs[p][step][2] for p in pick]
        real_outs = [trajs[p][step][1] for p in pick]
        policy_outs = [policy_trajs[p][step][1] for p in pick]

        # Verify array shapes
        real_outs = np.asarray(real_outs)
        policy_outs = np.asarray(policy_outs)
        assert policy_outs.shape == real_outs.shape
        rews = np.asarray(rews)
        assert rews.shape == (len(rews),)

        # True off-policy version of REINFORCE (!!!)
        grads = np.multiply((real_outs - policy_outs).T, rews.T).T

        if policy.n_states >= 1:
            grads = np.concatenate(
                (grads, [state_grads[p] for p in pick]),
                axis=1
            )

        grad_sum += policy.param_gradient_sum(inps, grads)
        grad_count += len(pick)

        # Backpropagation through time
        if step >= 1 and policy.n_states >= 1:
            grads = policy.input_gradients(inps, grads)
            grads = np.asarray(grads)[:,policy.n_inputs:]
            assert grads.shape == (len(pick), policy.n_states)
            grads = np.clip(grads, -state_grad_clip, state_grad_clip)
            for p, g in enumerate(grads):
                state_grads[p] = g

    return grad_sum / grad_count

def policy_gradient_stateless(trajs, *, policy):
    import numpy as np

    inps = []
    real_outs = []
    rews = []

    for t in trajs:
        for i, o, r in t:
            inps.append(i)
            real_outs.append(o)
            rews.append(r)

    inps = np.asarray(inps)
    real_outs = np.asarray(real_outs)
    rews = np.asarray(rews)

    policy_outs = policy.outputs(inps)
    policy_outs = np.asarray(policy_outs)

    assert policy_outs.shape == real_outs.shape
    assert rews.shape == (len(rews),)

    # True off-policy version of REINFORCE (!!!)
    grads = np.multiply((real_outs - policy_outs).T, rews.T).T

    grad_sum = policy.param_gradient_sum(inps, grads)
    return grad_sum / len(grads)
