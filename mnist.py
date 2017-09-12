#!/usr/bin/python3

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Mnist, Accuracy, PrintReward
from models import BasicNet, Softmax
from execute import policy_gradient
from optimizers import Momentum

def run():
    model = BasicNet(28*28, 128, "lrelu", 10)
    model = Softmax(model)

    world = Mnist()

    opt = Momentum(
        np.random.randn(model.n_params),
        lr=300.0,
        decay=0.9,
        print_info=True
    )

    for _ in range(200):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 128)
        grad = policy_gradient(model, trajs)
        opt.apply_gradient(grad)

    test_world = Accuracy(Mnist(test=True))
    PrintReward(test_world).trajectories(model, 5000)

if __name__ == "__main__":
    run()
