#!/usr/bin/python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Mnist, Accuracy, PrintReward, StochasticPolicy
from models import Input, Layer, Softmax
from execute import policy_gradient
from optimizers import Adams

def run():
    model = Input(28, 28)
    model = Layer(model, 128, "lrelu")
    model = Layer(model, 10)
    model = Softmax(model)

    train_world = PrintReward(
        StochasticPolicy(Accuracy(Mnist())),
        max_value=100.0
    )

    opt = Adams(
        np.random.randn(model.n_params),
        lr=0.00005
    )

    for i in range(600):
        model.load_params(opt.get_value())
        trajs = train_world.trajectories(model, 128)
        grad = policy_gradient(model, trajs)
        opt.apply_gradient(grad)

if __name__ == "__main__":
    run()
