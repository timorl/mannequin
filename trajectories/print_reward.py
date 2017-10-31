
from .get_rewards import get_rewards

def print_reward(trajs, *, max_value,
        label="Reward/episode: ",
        episode=None):
    import numpy as np

    if episode is None:
        episode = np.sum

    avg = np.mean(get_rewards(trajs, episode=episode))
    info = ("%%%d.2f" % len("-%.2f" % max_value)) % avg

    bar = max(0.0, min(1.0, abs(avg) / abs(max_value)))
    bar = int(round(bar * 50.0))
    if avg >= 0.0:
        info += " [" + "+" * bar + " " * (50 - bar) + "]"
    else:
        info += " [" + " " * (50 - bar) + "-" * bar + "]"

    print(label + info, flush=True)
