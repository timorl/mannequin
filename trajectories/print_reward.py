
from .get_rewards import get_rewards

def print_reward(trajs, *, max_value,
        label="Reward/episode:",
        episode=None):
    import numpy as np

    if episode is None:
        episode = np.sum

    avg = np.mean(get_rewards(trajs, episode=episode))

    info = "%s %10.2f" % (label, avg)
    bar = max(0.0, min(1.0, abs(avg) / abs(max_value)))
    bar = int(round(bar * 50.0))

    if avg >= 0.0:
        info += " [" + "+" * bar + " " * (50 - bar) + "]"
    else:
        info += " [" + " " * (50 - bar) + "-" * bar + "]"

    print(info)
