
from . import World

class Normalized(World):
    def __init__(self, inner, running=0.0):
        import sys
        import numpy as np

        avg = RunningMean(running)
        var = RunningMean(running)

        def trajectories(agent, n):
            trajs = inner.trajectories(agent, n)

            all_rewards = []
            for t in trajs:
                for o, a, r in t:
                    assert len(r) == 1
                    all_rewards.append(float(r[0]))

            avg.update(np.mean(all_rewards))
            var.update(np.mean(np.square(all_rewards - avg.get())))
            std = np.sqrt(var.get())

            if std < 0.000001:
                std = 1.0
                sys.stderr.write("Warning: all rewards are equal\n")

            for t in trajs:
                for o, a, r in t:
                    r[0] = (r[0] - avg.get()) / std

            return trajs

        self.trajectories = trajectories

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
