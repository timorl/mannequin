
def process_rewards(trajs, *, process):
    import numpy as np

    def replaced_rewards(traj):
        rew = np.array([float(r) for o, a, r in traj])
        rew.setflags(write=False)

        rew = np.asarray(process(rew), dtype=np.float32)
        assert rew.shape == (len(traj),)

        return [(o, a, r) for (o, a, _), r in zip(traj, rew)]

    return [replaced_rewards(t) for t in trajs]
