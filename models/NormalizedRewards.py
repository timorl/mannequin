
from .BaseWrapper import BaseWrapper

class NormalizedRewards(BaseWrapper):
    def __init__(self, inner):
        import sys
        import numpy as np

        # Summing over trajectories needs to make sense
        assert inner.rew_shape == (1,)

        def param_gradient(trajs):
            all_rewards = []
            all_weights = []
            for t in trajs:
                for o, a, r in t:
                    assert len(r) == 1
                    all_rewards.append(r[0])
                    all_weights.append(1.0 / len(t))

            avg = np.average(all_rewards, weights=all_weights)
            std = np.sqrt(np.average(
                np.square(all_rewards - avg),
                weights=all_weights
            ))
            if std < 0.000001:
                std = 1.0
                sys.stderr.write("Warning: all rewards are the same\n")

            trajs = [
                [(o, a, [(r[0] - avg) / std]) for o, a, r in t]
                for t in trajs
            ]

            return inner.param_gradient(trajs)

        super().__init__(
            inner,
            param_gradient=param_gradient
        )
