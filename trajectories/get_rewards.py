
def get_rewards(trajs, *,
        reward=lambda x: x,
        episode=lambda x: x):
    return [
        episode([reward(float(r)) for o, a, r in t])
        for t in trajs
    ]
