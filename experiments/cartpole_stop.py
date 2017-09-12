#!/usr/bin/python3

import os
import sys
import numpy as np
import gym

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, Normalized, Future, PrintReward
from models import Input, Layer, Softmax, RandomChoice
from execute import policy_gradient
from optimizers import Adams

def make_env():
    env = gym.make("CartPole-v1")
    orig_step = env.step
    def step(action):
        env.unwrapped.steps_beyond_done = None
        o, r, _, i = orig_step(action)
        return o, -abs(o[0]), False, i
    env.step = step
    return env

def run():
    model = Input(4)
    model = Layer(model, 128, "lrelu")
    model = Layer(model, 2)
    model = Softmax(model)
    model = RandomChoice(model)

    world = Gym(make_env, max_steps=500)

    train_world = Normalized(
        Future(
            PrintReward(world, max_value=5000.0),
            horizon=50
        )
    )

    opt = Adams(
        np.random.randn(model.n_params) * 0.1,
        lr=0.00002
    )

    for _ in range(50):
        model.load_params(opt.get_value())
        trajs = train_world.trajectories(model, 16)
        grad = policy_gradient(model, trajs)
        opt.apply_gradient(grad)

    while True:
        world.render(model)

if __name__ == "__main__":
    run()
