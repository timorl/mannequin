#!/usr/bin/env python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, StochasticPolicy
from models import Input, Affine, Softmax, LReLU
from trajectories import normalize, discount, policy_gradient, print_reward, get_rewards
from optimizers import Adams

def train(world, model):
    opt = Adams(
        np.random.randn(model.n_params) * 0.1,
        lr=0.0001,
        memory=0.8
    )

    while True:
        model.load_params(opt.get_value())

        trajs = world.trajectories(model, 16)
        print_reward(trajs, max_value=500)

        if np.mean(get_rewards(trajs, episode=np.sum)) >= 498:
            return opt.get_value()

        trajs = discount(trajs, horizon=500)
        trajs = normalize(trajs)

        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

def run():
    world = StochasticPolicy(Gym("CartPole-v1"))

    model = Input(4)
    model = Affine(model, 64)
    model = LReLU(model)
    model = Affine(model, 2)
    model = Softmax(model)

    if len(sys.argv) >= 2:
        params = np.load(sys.argv[1])
    else:
        params = train(world, model)
        np.save("__cartpole.npy", params)

    model.load_params(params)
    world.render(model)

if __name__ == "__main__":
    run()
