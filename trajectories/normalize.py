
def normalize(trajs):
    import numpy as np

    all_rewards = []
    for t in trajs:
        for o, a, r in t:
            all_rewards.append(float(r))
    all_rewards = np.asarray(all_rewards)

    if len(all_rewards) < 2:
        raise ValueError("Cannot normalize a single reward")

    avg = np.mean(all_rewards)
    std = np.std(all_rewards)

    if std < 0.000001:
        std = 1.0
        import sys
        sys.stderr.write("Warning: all rewards are equal\n")

    return [
        [(o, a, (r - avg) / std) for o, a, r in t]
        for t in trajs
    ]
