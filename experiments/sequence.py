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
from models import Constant, Softmax, Input, Layer, Memory
from trajectories import policy_gradient, cross_entropy, print_reward
from optimizers import Adam

def train(world, model, *, lr):
    opt = Adam(
        np.random.randn(model.n_params),
        lr=lr,
        memory=0.9
    )

    for _ in range(20):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 100)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

        trajs = cross_entropy(trajs, model=model)
        print_reward(trajs, episode=np.mean,
            label="Surprise/byte:", max_value=8.0)

def run():
    world = Bytes(b"abcdabcdabcdabcd", max_steps=4)

    print("\nConstant model\n")
    model = Constant(256)
    model = Softmax(model)
    train(world, model, lr=2.0)

    print("\nMemory\n")
    model = Input(2, 256)
    model = Layer(model, 256)
    model = Memory(model)
    model = Softmax(model)
    train(world, model, lr=10.0)

if __name__ == "__main__":
    run()
