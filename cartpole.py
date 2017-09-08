#!/usr/bin/python3

import os
if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

import numpy as np
from worlds import Gym
from models import (
    BasicNet,
    Softmax,
    Reinforce,
    SampleFrom,
    NormalizedRewards,
    EpisodeSum
)

world = Gym("CartPole-v1")

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def train(model):
    params = np.random.randn(model.n_params)
    model.load_params(params)
    first_norm = None
    for _ in range(30):
        trajs = world.trajectories(model, 16)
        grad = model.param_gradient(trajs)
        grad_norm = norm(grad)
        if first_norm is None:
            first_norm = grad_norm
        update = grad * (4.0 / max(first_norm, grad_norm * 0.5))
        print("Update norm: %9.6f" % norm(update))
        params += update
        model.load_params(params)
    return params

def score(model):
    rew_sum = 0.0
    for _ in range(2):
        for t in world.trajectories(model, n=16):
            for o, a, r in t:
                rew_sum += np.mean(r)
    return rew_sum / 32.0

def run():
    model = BasicNet([4, "relu", 32, "relu", 2])
    model = Softmax(model)
    model = Reinforce(model)
    model = SampleFrom(model)
    model = NormalizedRewards(model)
    model = EpisodeSum(model)

    best_params = None
    best_score = 0.0

    for _ in range(3):
        params = train(model)
        s = score(model)
        print("Reward/episode:           %9.5f" % s)

        if s > best_score:
            best_score = s
            best_params = params

    print("Best of 3 reward/episode: %9.5f" % best_score)

    if "DEBUG" in os.environ:
        model.load_params(best_params)
        for _ in range(5):
            world.render(model)
    else:
        assert best_score > 50.0

if __name__ == "__main__":
    run()
