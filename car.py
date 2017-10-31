#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, ActionNoise, Cache
from models import Input, Affine, Softmax, LReLU
from trajectories import normalize, discount, policy_gradient, print_reward, replace_rewards, get_rewards
from optimizers import Adam

def plot_trajs(file_name, trajs, *,
        xs=lambda t: [o[0] for o, a, r in t],
        ys=lambda t: [o[1] for o, a, r in t],
        color=lambda t: "b"):
    plt.clf()
    plt.grid()
    plt.gcf().axes[0].set_xlim([-1.25, 0.5])
    plt.gcf().axes[0].set_ylim([-0.075, 0.075])
    for t in trajs:
        plt.plot(xs(t), ys(t), color=color(t), alpha=0.1)
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def supervised(trajs, *, label):
    return [
        [(o, label(o, a, r), 1.0) for o, a, r in t]
        for t in trajs
    ]

def curiosity(world, agent, classifier, memory, log_path):
    agent_opt = Adam(
        np.random.randn(agent.n_params),
        lr=0.05,
        memory=0.5
    )

    classifier_opt = Adam(
        np.random.randn(classifier.n_params),
        lr=0.1,
        memory=0.5
    )

    for episode in range(50):
        agent.load_params(agent_opt.get_value())
        classifier.load_params(classifier_opt.get_value())

        agent_trajs = world.trajectories(agent, 8)

        classifier_trajs = (
            supervised(
                memory.trajectories(None, 8),
                label=lambda *_: [1, 0]
            )
            + supervised(
                agent_trajs,
                label=lambda *_: [0, 1]
            )
        )

        grad = policy_gradient(classifier_trajs, policy=classifier)
        classifier_opt.apply_gradient(grad)

        agent_trajs = replace_rewards(
            agent_trajs,
            model=classifier,
            reward=lambda o: o[1],
            episode=np.max
        )

        print_reward(agent_trajs, max_value=1.0,
            episode=np.mean, label="Curiosity reward: ")

        plot_trajs(
            log_path + "_%04d.png" % (episode + 1),
            classifier_trajs,
            color=lambda t: ["b", "r"][np.argmax(t[0][1])]
        )

        if np.mean(get_rewards(agent_trajs, episode=np.mean)) > 0.95:
            return agent_opt.get_value()

        agent_trajs = normalize(agent_trajs)
        grad = policy_gradient(agent_trajs, policy=agent)
        agent_opt.apply_gradient(grad)

    return agent_opt.get_value()

def build_agent():
    model = Input(2)
    model = Affine(model, 32)
    model = LReLU(model)
    model = Affine(model, 1)
    return model

def build_classifier():
    model = Input(2)
    model = Affine(model, 16)
    model = LReLU(model)
    model = Affine(model, 2)
    model = Softmax(model)
    return model

def load_or_compute(file_name, compute):
    if os.path.exists(file_name):
        print("Loading from '%s'... " % file_name, end="", flush=True)
        data = np.load(file_name)
    else:
        data = compute()
        print("Saving to '%s'... " % file_name, end="", flush=True)
        np.save(file_name, data)
    print("OK", flush=True)
    return data

def solve(world):
    world = ActionNoise(world, stddev=0.1)
    agent = build_agent()
    classifier = build_classifier()
    memory = Cache()
    agent_id = 0

    print("Random agent")
    params = load_or_compute(
        "__car_%02d.npy" % agent_id,
        lambda: np.random.randn(agent.n_params)
    )

    agent.load_params(params)
    memory.add_trajectory(*world.trajectories(agent, 16))

    while True:
        agent_id += 1

        print("Curious agent %d" % agent_id)
        params = load_or_compute(
            "__car_%02d.npy" % agent_id,
            lambda: curiosity(world, agent, classifier, memory,
                "__car_%02d" % agent_id)
        )

        agent.load_params(params)
        trajs = world.trajectories(agent, 16)

        print_reward(trajs, max_value=100.0, label="Real reward:    ")
        if np.mean(get_rewards(trajs)) > 20.0:
            return params

        memory.add_trajectory(*trajs)

def run():
    world = Gym("MountainCarContinuous-v0")
    agent = build_agent()

    if len(sys.argv) >= 2:
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        agent.load_params(solve(world))
        world.render(agent)

if __name__ == "__main__":
    run()
