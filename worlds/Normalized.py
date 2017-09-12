
from .BaseWorld import BaseWorld

class Normalized(BaseWorld):
    def __init__(self, inner, *, running=0.0):
        import sys
        import numpy as np

        avg = RunningMean(running)
        var = RunningMean(running)

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)

            all_rewards = []
            for t in trajs:
                for o, a, r in t:
                    all_rewards.append(float(r))
            all_rewards = np.asarray(all_rewards)

            avg.update(np.mean(all_rewards))
            avg_val = avg.get()

            var.update(np.mean(np.square(all_rewards - avg_val)))
            stddev = np.sqrt(var.get())

            if stddev < 0.000001:
                stddev = 1.0
                sys.stderr.write("Warning: all rewards are equal\n")

            return [
                [(o, a, (r - avg_val) / stddev) for o, a, r in t]
                for t in trajs
            ]

        self.trajectories = trajectories
        self.render = inner.render

class RunningMean(object):
    def __init__(self, decay):
        decay = float(decay)
        assert decay >= 0.0
        assert decay < 1.0

        biased_mean = 0.0
        decay_power = 1.0

        def update(value):
            nonlocal biased_mean, decay_power

            biased_mean = biased_mean * decay + value * (1.0 - decay)
            decay_power *= decay

        self.get = lambda: biased_mean / (1.0 - decay_power)
        self.update = update
