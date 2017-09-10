#!/usr/bin/python3

import numpy as np
from worlds import Gym, Normalized, Future
from models import BasicNet, Softmax, OffPolicy, RandomChoice
from optimizers import Adam

world = Gym("CartPole-v1")

def train(model):
    train_world = Normalized(
        Future(world, horizon=500)
    )

    opt = Adam(
        np.random.randn(model.n_params) * 0.1,
        lr=0.00015,
        decay=0.9
    )

    for _ in range(30):
        grads = []

        for params in opt.get_requests():
            model.load_params(params)

            trajs = train_world.trajectories(model, 16)
            ep_len = np.mean([len(t) for t in trajs])

            print("%30s | %s %.1f" % (opt.get_info(),
                "=" * int(round(0.1 * ep_len)), ep_len))

            grads.append(model.param_gradient(trajs))

        opt.feed_gradients(grads)

    model.load_params(opt.get_best_value())

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
