
from .BaseWrapper import BaseWrapper

class EpisodeSum(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        # Summing over trajectories needs to make sense
        assert inner.rew_shape == (1,)

        def param_gradient(trajs):
            traj_rewards = np.zeros(len(trajs), dtype=np.float32)
            for i, traj in enumerate(trajs):
                rews = np.array([r for o, a, r in traj])
                rews = rews.reshape(-1)
                assert len(rews) == len(traj)
                traj_rewards[i] = np.sum(rews)

            trajs = [
                [(o, a, [traj_rewards[i]]) for o, a, _ in traj]
                for i, traj in enumerate(trajs)
            ]

            return inner.param_gradient(trajs)

        super().__init__(
            inner,
            param_gradient=param_gradient
        )
