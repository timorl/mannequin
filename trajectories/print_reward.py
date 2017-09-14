
from .episode_sum import episode_sum

def print_reward(trajs, *, max_value, label="Reward/episode:"):
    import numpy as np

    avg = np.mean([t[0][2] for t in episode_sum(trajs)])

    info = "%s %10.2f" % (label, avg)
    bar = max(0.0, min(1.0, abs(avg) / abs(max_value)))
    bar = int(round(bar * 50.0))

    if avg >= 0.0:
        info += " [" + "+" * bar + " " * (50 - bar) + "]"
    else:
        info += " [" + " " * (50 - bar) + "-" * bar + "]"

    print(info)
