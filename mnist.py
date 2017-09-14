#!/usr/bin/python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Mnist
from models import Input, Layer, Conv2d, Maxpool, Softmax
from trajectories import policy_gradient, accuracy, print_reward
from optimizers import Adam

def run():
    model = Input(28, 28)
    model = Conv2d(model, size=5, channels=32)
    model = Maxpool(model, size=2)
    model = Conv2d(model, size=5, channels=64)
    model = Maxpool(model, size=2)
    model = Layer(model, 128, "lrelu")
    model = Layer(model, 10)
    model = Softmax(model)

    world = Mnist()

    opt = Adam(
        np.random.randn(model.n_params),
        lr=0.1
    )

    for i in range(600):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 128)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

        trajs = accuracy(trajs, model=model, percent=True)
        print_reward(trajs, max_value=100, label="Train accuracy:")

    print()
    trajs = Mnist(test=True).trajectories(None, 5000)
    trajs = accuracy(trajs, model=model, percent=True)
    print_reward(trajs, max_value=100, label="Test accuracy: ")

if __name__ == "__main__":
    run()
