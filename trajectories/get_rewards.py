
def get_rewards(trajs, *,
        reward=lambda x: x,
        episode=None):
    import numpy as np

    if episode is None:
        episode = np.sum

    return [
        episode([reward(float(r)) for o, a, r in t])
        for t in trajs
    ]
