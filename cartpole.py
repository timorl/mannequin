#!/usr/bin/python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, StochasticPolicy
from models import Input, Layer, Softmax
from trajectories import normalize, discount, policy_gradient, print_reward
from optimizers import Adams

def run():
    model = Input(4)
    model = Layer(model, 64, "lrelu")
    model = Layer(model, 2)
    model = Softmax(model)

    world = StochasticPolicy(Gym("CartPole-v1"))

    opt = Adams(
        np.random.randn(model.n_params) * 0.1,
        lr=0.0001,
        mean_decay=0.8
    )

    for _ in range(20):
        model.load_params(opt.get_value())

        trajs = world.trajectories(model, 16)
        print_reward(trajs, max_value=500)

        trajs = discount(trajs, horizon=500)
        trajs = normalize(trajs)

        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

    while True:
        world.render(model)

if __name__ == "__main__":
    run()
