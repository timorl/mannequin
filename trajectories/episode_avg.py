
def episode_avg(trajs):
    import numpy as np

    def process(traj):
        all_rew = [float(r) for o, a, r in traj]
        avg_rew = np.mean(all_rew)
        return [(o, a, avg_rew) for o, a, r in traj]

    return [process(t) for t in trajs]
