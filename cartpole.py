#!/usr/bin/python3

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, Normalized, Future, PrintReward
from models import BasicNet, Softmax, OffPolicy, RandomChoice
from optimizers import Adam

def run():
    model = BasicNet([4, "lrelu", 64, "lrelu", 2])
    model = Softmax(model)
    model = OffPolicy(model)
    model = RandomChoice(model)

    world = Gym("CartPole-v1")

    train_world = Normalized(
        Future(
            PrintReward(world, max_value=500.0),
            horizon=500
        )
    )

    opt = Adam(
        np.random.randn(model.n_params) * 0.1,
        lr=0.0001,
        decay=0.8,
        square=True
    )

    for _ in range(30):
        model.load_params(opt.get_value())
        trajs = train_world.trajectories(model, 16)
        grad = model.param_gradient(trajs)
        opt.apply_gradient(grad)

    for _ in range(5):
        world.render(model)

if __name__ == "__main__":
    run()
