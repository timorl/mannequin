
def retrace(trajs, *, model,
        max_steps=-1,
        with_states=False):
    import numpy as np

    outputs = [[] for _ in trajs]
    states = [None] * len(trajs)
    step = 0

    while step != max_steps:
        # Find trajectories that are still active
        pick = [i for i, t in enumerate(trajs) if len(t) > step]
        if len(pick) < 1:
            break

        # Get model predictions
        picked_inputs = []
        for i in pick:
            inp = np.asarray(trajs[i][step][0])
            if model.n_inputs == 0:
                assert states[i] is None
                picked_inputs.append(None)
            else:
                assert inp.shape == (model.n_inputs,)
                if states[i] is not None:
                    inp = np.concatenate((inp, states[i]))
                picked_inputs.append(inp)

        picked_outputs = model.outputs(picked_inputs)
        picked_outputs = np.asarray(picked_outputs)
        assert len(picked_outputs) == len(pick)

        # Save outputs in a way that matches shapes of trajectories
        for i, o in zip(pick, picked_outputs):
            o = np.reshape(o, -1)
            n = model.n_outputs
            if len(o) > n:
                # Preserve state
                states[i] = o[n:]
                if not with_states:
                    o = o[:n]
            outputs[i].append(o)

        step += 1

    return outputs
