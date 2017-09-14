
def episode_sum(trajs):
    import numpy as np

    def process(traj):
        all_rew = [float(r) for o, a, r in traj]
        rew_sum = np.sum(all_rew)
        return [(o, a, rew_sum) for o, a, r in traj]

    return [process(t) for t in trajs]
