#!/usr/bin/python3

import os
if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

import numpy as np
from worlds import Mnist, Accuracy
from models import BasicNet, Softmax

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def train(model):
    world = Mnist()
    params = np.random.randn(model.n_params)
    model.load_params(params)
    for _ in range(200):
        trajs = world.trajectories(model, 128)
        update = model.param_gradient(trajs) * 300.0
        print("Update norm: %12.6f" % norm(update))
        params += update
        model.load_params(params)
    return params

def score(model, n=5000):
    world = Accuracy(Mnist(test=True))
    rew_sum = 0.0
    for t in world.trajectories(model, n):
        for o, a, r in t:
            rew_sum += np.mean(r)
    return rew_sum / n

def run():
    model = BasicNet([28*28, "relu", 128, "relu", 10])
    model = Softmax(model)

    train(model)
    s = score(model)

    print("Accuracy on test:         %9.2f%%" % (100.0 * s))
    assert s > 0.9
    assert s < 1.0

if __name__ == "__main__":
    run()
