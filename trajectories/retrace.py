
def retrace(trajs, *, model, with_inputs=False):
    if model.n_states <= 0:
        return retrace_stateless(trajs, model=model,
            with_inputs=with_inputs)

    import numpy as np

    outputs = [[] for _ in trajs]
    states = [np.zeros(model.n_states)] * len(trajs)
    step = 0

    while True:
        # Find trajectories that are still active
        pick = [p for p, t in enumerate(trajs) if len(t) > step]
        if len(pick) < 1:
            break

        # Get model predictions
        picked_inputs = []
        for p in pick:
            inp = np.asarray(trajs[p][step][0])
            if model.n_inputs == 0:
                # Ignore the actual input from the trajectory
                picked_inputs.append(states[p])
            else:
                assert inp.shape == (model.n_inputs,)
                if len(states[p]) >= 1:
                    inp = np.concatenate((inp, states[p]))
                picked_inputs.append(inp)

        picked_outputs = model.outputs(picked_inputs)
        picked_outputs = np.asarray(picked_outputs)
        assert len(picked_outputs) == len(pick)

        # Save outputs in a way that matches shapes of trajectories
        for p, i, o in zip(pick, picked_inputs, picked_outputs):
            o = np.reshape(o, -1)
            if model.n_states >= 1:
                # Preserve state
                states[p] = o[model.n_outputs:]
                o = o[:model.n_outputs]
            assert len(o) == model.n_outputs
            assert len(states[p]) == model.n_states
            if with_inputs:
                outputs[p].append((i, o))
            else:
                outputs[p].append(o)

        step += 1

    return outputs

def retrace_stateless(trajs, *, model, with_inputs=False):
    import numpy as np

    all_inputs = []
    for t in trajs:
        for i, o, r in t:
            all_inputs.append(i)
    all_inputs = np.asarray(all_inputs)

    all_outputs = model.outputs(all_inputs)
    all_outputs = np.asarray(all_outputs)
    assert len(all_outputs) == len(all_inputs)

    if with_inputs:
        all_outputs = zip(all_inputs, all_outputs)
    all_outputs = all_outputs.__iter__()

    return [
        [next(all_outputs) for _ in t]
        for t in trajs
    ]
