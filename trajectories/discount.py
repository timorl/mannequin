
from .replace_rewards import replace_rewards

def discount(trajs, *, horizon):
    multiplier = 1.0 - (1.0 / horizon)

    def process(rew):
        rew_sum = 0.0
        reversed_out = []

        for r in reversed(rew):
            rew_sum = rew_sum * multiplier + float(r)
            reversed_out.append(rew_sum)

        return list(reversed(reversed_out))

    return replace_rewards(trajs, episode=process)
