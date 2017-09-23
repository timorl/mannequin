#!/usr/bin/python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Bytes
from models import Constant, Softmax
from trajectories import policy_gradient, cross_entropy, print_reward
from optimizers import Adam

def run():
    model = Constant(256)
    model = Softmax(model)
    world = Bytes(b"abcdabcdabcdabcd", max_steps=4)

    opt = Adam(
        np.random.randn(model.n_params),
        lr=1.0,
        memory=0.9
    )

    for _ in range(20):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 100)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

        trajs = cross_entropy(trajs, model=model, negative=True)
        print_reward(trajs, max_value=32.0)

if __name__ == "__main__":
    run()
