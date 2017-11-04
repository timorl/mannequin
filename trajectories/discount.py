
from .replace_rewards import replace_rewards

def discount(trajs, *,
        horizon,
        combine=lambda a, b: a+b):
    multiplier = 1.0 - (1.0 / horizon)

    def process(rew):
        rew_sum = 0.0
        reversed_out = []

        for r in reversed(rew):
            rew_sum *= multiplier
            rew_sum = float(combine(rew_sum, float(r)))
            reversed_out.append(rew_sum)

        return list(reversed(reversed_out))

    return replace_rewards(trajs, episode=process)
