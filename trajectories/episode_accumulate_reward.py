
import numpy as np

def episode_accumulate_reward(trajs, accumulator):

    def process(traj):
        all_rew = [float(r) for o, a, r in traj]
        rew_sum = accumulator(all_rew)
        return [(o, a, rew_sum) for o, a, r in traj]

    return [process(t) for t in trajs]
