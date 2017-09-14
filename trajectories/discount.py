
def discount(trajs, *, horizon):
    import numpy as np

    multiplier = 1.0 - (1.0 / horizon)

    def process(traj):
        rew_sum = 0.0
        reversed_out = []

        for o, a, r in reversed(traj):
            rew_sum = rew_sum * multiplier + float(r)
            reversed_out.append((o, a, rew_sum))

        return list(reversed(reversed_out))

    return [process(t) for t in trajs]
