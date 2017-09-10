
from . import World

class PrintReward(World):
    def __init__(self, inner, max_value=None):
        import sys
        import numpy as np

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)

            rew_sum = 0.0
            for t in trajs:
                for o, a, r in t:
                    assert len(r) == 1
                    rew_sum += float(r[0])
            rew_sum /= len(trajs)

            info = "Reward/episode: %10.2f" % rew_sum

            if max_value is not None:
                bar = max(-1.0, min(1.0, rew_sum / max_value))
                bar = int(round(bar * 50.0))
                if bar >= 0.0:
                    info += " [" + "+" * bar + " " * (50 - bar) + "]"
                else:
                    info += " [" + " " * (50 + bar) + "-" * -bar + "]"

            print(info)
            return trajs

        self.trajectories = trajectories
