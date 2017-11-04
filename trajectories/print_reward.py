
from .get_rewards import get_rewards

def print_reward(trajs, *, max_value,
        label="Reward/episode: ",
        after="",
        episode=None):
    import numpy as np

    if episode is None:
        episode = np.sum

    reward = np.mean(get_rewards(trajs, episode=episode))
    info = ("%%%d.2f" % len("-%.2f" % max_value)) % reward

    bar = max(0.0, min(1.0, abs(reward) / abs(max_value)))
    bar = int(round(bar * 50.0))
    if reward >= 0.0:
        info += " [" + "+" * bar + " " * (50 - bar) + "]"
    else:
        info += " [" + " " * (50 - bar) + "-" * bar + "]"

    print(label + info + after, flush=True)
