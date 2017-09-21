
def retrace(trajs, *, model,
        max_steps=-1,
        with_states=False):
    import numpy as np

    results = [[] for _ in trajs]
    states = [None] * len(results)
    step = 0

    while step != max_steps:
        # Find trajectories that are still active
        pick = [n for n, t in enumerate(trajs) if len(t) > step]
        if len(pick) < 1:
            break

        # Get model predictions
        pick_inputs = [trajs[n][step][0] for n in pick]
        pick_states = [states[n] for n in pick]
        pick_states, model_outs = model.step(pick_states, pick_inputs)
        model_outs = np.asarray(model_outs)
        assert len(model_outs) == len(pick)

        # Save outputs in a way that matches shapes of trajectories
        for n, s, o in zip(pick, pick_states, model_outs):
            results[n].append((s, o) if with_states else o)
            states[n] = s

        step += 1

    return results
