
from .BaseWorld import BaseWorld

class PrintReward(BaseWorld):
    def __init__(self, inner, *,
            max_value=None, label="Reward/episode:"):
        import sys
        import numpy as np

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)

            rew_sum = 0.0
            for t in trajs:
                for o, a, r in t:
                    rew_sum += float(r)
            rew_sum /= len(trajs)

            info = "%s %10.2f" % (label, rew_sum)

            if max_value is not None:
                bar = max(0.0, min(1.0, abs(rew_sum) / abs(max_value)))
                bar = int(round(bar * 50.0))
                if rew_sum >= 0.0:
                    info += " [" + "+" * bar + " " * (50 - bar) + "]"
                else:
                    info += " [" + " " * (50 - bar) + "-" * bar + "]"

            print(info)
            return trajs

        self.trajectories = trajectories
        self.render = inner.render
