
def get_reward(trajs, *, episode, episodes=lambda x: x):
    return episodes([
        episode([float(r) for o, a, r in t])
        for t in trajs
    ])
