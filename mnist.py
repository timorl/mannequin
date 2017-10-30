#!/usr/bin/env python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Mnist
from models import Input, Affine, LReLU, Conv2d, Maxpool, Softmax
from trajectories import policy_gradient, accuracy, print_reward, get_rewards
from optimizers import Adams

def train(model):
    world = Mnist()

    opt = Adams(
        np.random.randn(model.n_params),
        lr=0.06,
        power=1.1
    )

    for i in range(600):
        model.load_params(
            opt.get_value()
            + np.random.randn(model.n_params) * 0.01
        )

        trajs = world.trajectories(None, 256)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

        if i % 20 == 19:
            print("%4d) " % (i+1), flush=True, end="")
            trajs = world.trajectories(None, 2000)
            trajs = accuracy(trajs, model=model, percent=True)
            print_reward(trajs, max_value=100, label="Train accuracy:")

    return opt.get_value()

def run():
    model = Input(28, 28)
    model = Conv2d(model, size=3, channels=8)
    model = LReLU(model)
    model = Maxpool(model, size=2)
    model = Conv2d(model, size=5, channels=16)
    model = LReLU(model)
    model = Maxpool(model, size=2)
    model = Affine(model, 128)
    model = LReLU(model)
    model = Affine(model, 10)
    model = Softmax(model)

    if len(sys.argv) >= 2:
        params = np.load(sys.argv[1])
    else:
        params = train(model)
        np.save("__mnist.npy", params)

    model.load_params(params)
    test_world = Mnist(test=True)
    trajs = test_world.trajectories(None, 5000)
    trajs = accuracy(trajs, model=model, percent=True)
    print_reward(trajs, max_value=100, label="Test accuracy:")

if __name__ == "__main__":
    run()
