#!/usr/bin/python3

import numpy as np
from worlds import Gym, Normalized, Future
from models import BasicNet, Softmax, OffPolicy, RandomChoice

world = Gym("CartPole-v1")

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def train(model):
    train_world = Normalized(Future(world, horizon=500))

    params = np.random.randn(model.n_params) * 0.1
    model.load_params(params)

    lr = 0.0
    update = 0.0
    running_ep_len = 20.0

    while True:
        model.load_params(params)

        trajs = train_world.trajectories(model, 16)
        grad = model.param_gradient(trajs)

        ep_len = np.mean([len(t) for t in trajs])
        if ep_len >= 400.0:
            break
        running_ep_len = running_ep_len * 0.8 + ep_len * 0.2

        print("Update norm: %9.6f  [%s]  ep=%.1f"
            % (norm(update) * lr,
                "=" * int(round(0.1 * ep_len)), ep_len), end="")

        if ep_len > running_ep_len + 30.0:
            update = 0.0
            running_ep_len = ep_len
            lr = 700.0 / running_ep_len
            print("  lr=%.1f  [BOUNCE]" % lr)
        else:
            lr = 700.0 / running_ep_len
            print("  lr=%.1f" % lr)

        update = update * 0.98 + grad
        params += update * lr

    return params

def score(model):
    rew_sum = 0.0
    for _ in range(2):
        for t in world.trajectories(model, n=16):
            for o, a, r in t:
                rew_sum += np.mean(r)
    return rew_sum / 32.0

def run():
    model = BasicNet([4, "lrelu", 64, "lrelu", 2])
    model = Softmax(model)
    model = OffPolicy(model)
    model = RandomChoice(model)

    params = train(model)
    s = score(model)
    print("Reward/episode: %.2f" % s)

    for _ in range(5):
        world.render(model)

if __name__ == "__main__":
    run()
