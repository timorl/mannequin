#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, ActionNoise
from models import Input, Affine, Softmax, LReLU
from trajectories import (normalize, discount, policy_gradient,
    print_reward, get_rewards, replace_rewards, retrace)
from optimizers import Adam, Adams

def build_agent():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 1)
    return model

def build_oracle():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 2)
    return model

def save_plot(file_name, trajs, predictions):
    import matplotlib.pyplot as plt
    from matplotlib import collections as coll
    lines = []
    for t, t_b in zip(trajs[:1], predictions):
        pos = np.array([0.25, 0.0])
        pos_b = np.array([0.75, 0.0])
        for (o1, delta, r), delta_b in zip(t, t_b):
            lines.append([pos, pos + delta*[1.0, 10.0]])
            lines.append([pos_b, pos_b + delta_b*[1.0, 10.0]])
            pos = pos + [0.0, 0.01]
            pos_b = pos_b + [0.0, 0.01]
    lc = coll.LineCollection(lines, linewidths=1)
    plt.clf()
    plt.grid()
    plt.gcf().axes[0].set_ylim([-0.3, 2.3])
    plt.gcf().axes[0].set_xlim([0.0, 1.0])
    plt.gcf().axes[0].add_collection(lc)
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def curiosity(world):
    world = ActionNoise(world, stddev=0.1)

    log_dir = "__oracle"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    agent = build_agent()
    agent_opt = Adams(
        np.random.randn(agent.n_params),
        lr=0.00015,
        memory=0.5
    )

    oracle = build_oracle()
    oracle_opt = Adam(
        np.random.randn(oracle.n_params) * 0.01,
        lr=0.05,
        memory=0.95
    )

    for episode in range(1000):
        agent.load_params(agent_opt.get_value())
        oracle.load_params(oracle_opt.get_value())

        agent_trajs = world.trajectories(agent, 4)

        oracle_trajs = [
            [
                (o1, o2-o1, 1.0)
                for (o1, a1, r1), (o2, a2, r2)
                in zip(t, t[10:])
            ]
            for t in agent_trajs
        ]

        predictions = retrace(oracle_trajs, model=oracle)

        save_plot(
            log_dir + "/%04d.png" % (episode + 1),
            oracle_trajs, predictions
        )

        score_trajs = [
            [
                (None, None, -np.sqrt(np.mean(np.square(delta - delta_b))))
                for (o1, delta, r), delta_b in zip(t, t_b)
            ]
            for t, t_b in
                zip(oracle_trajs, predictions)
        ]
        print_reward(score_trajs, episode=np.mean, max_value=0.01)

        grad = policy_gradient(oracle_trajs, policy=oracle)
        oracle_opt.apply_gradient(grad)

def run():
    world = Gym("MountainCarContinuous-v0", max_steps=200)

    if len(sys.argv) >= 2:
        agent = build_agent()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        curiosity(world)

if __name__ == "__main__":
    run()
