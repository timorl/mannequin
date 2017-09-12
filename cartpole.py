#!/usr/bin/python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, Normalized, Future, PrintReward
from models import Input, Layer, Softmax, RandomChoice
from execute import policy_gradient
from optimizers import Adams

def run():
    model = Input(4)
    model = Layer(model, 64, "lrelu")
    model = Layer(model, 2)
    model = Softmax(model)
    model = RandomChoice(model)

    world = Gym("CartPole-v1")

    train_world = Normalized(
        Future(
            PrintReward(world, max_value=500.0),
            horizon=500
        )
    )

    opt = Adams(
        np.random.randn(model.n_params) * 0.1,
        lr=0.0001,
        mean_decay=0.8
    )

    for _ in range(30):
        model.load_params(opt.get_value())
        trajs = train_world.trajectories(model, 16)
        grad = policy_gradient(model, trajs)
        opt.apply_gradient(grad)

    while True:
        world.render(model)

if __name__ == "__main__":
    run()
