
from .retrace import retrace

def replace_rewards(trajs, *,
        reward=lambda x:x,
        episode=lambda x:x,
        model=None):
    import numpy as np

    def make_traj(traj, rew):
        rew = np.asarray(rew, dtype=np.float32)
        assert rew.shape == (len(traj),)

        rew.setflags(write=False)
        rew = episode(rew)

        rew = np.asarray(rew, dtype=np.float32)
        rew = np.broadcast_to(rew, (len(traj),))
        assert rew.shape == (len(traj),)

        rew.setflags(write=False)
        return [(o, a, r) for (o, a, _), r in zip(traj, rew)]

    if model is None:
        return [
            make_traj(t, [float(reward(float(r))) for _, _, r in t])
            for t in trajs
        ]
    else:
        return [
            make_traj(t, [float(reward(o)) for o in outs])
            for t, outs in zip(trajs, retrace(trajs, model=model))
        ]
