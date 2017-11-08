#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy, BaseWorld, ActionNoise, Cache
from models import Input, Affine, LReLU, Softmax, Tanh
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, get_rewards, replace_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

AGENT_VARIATIONS=5
TRAJS_PER_VARIATION=6
EPISODE_DELAY=5

class Curiosity(BaseWorld):
    def __init__(self, inner, *, classifier, for_classifier=lambda x: x, plot=None):
        history = Cache(AGENT_VARIATIONS * TRAJS_PER_VARIATION * EPISODE_DELAY)
        classOpt = None

        def tag_traj(traj, tag):
            return [(t[0], tag, t[2]) for t in traj]

        def reset_classifier():
            nonlocal classOpt
            classOpt = Adam(
                np.random.randn(classifier.n_params) * 1.,
                lr=0.06,
                memory=0.9,
            )

        def trajectories(agents, n):
            if classOpt is None:
                reset_classifier()
                history.add_trajectories(inner.trajectories(agents[0], n))

            oldTrajs, innerTrajs = [], []
            for agent in agents:
                oldTrajs += history.trajectories(None, n)
                innerTrajs += inner.trajectories(agent, n)

            history.add_trajectories(innerTrajs)

            trajsForClass = for_classifier(
                [tag_traj(traj, [1,0]) for traj in oldTrajs]
                + [tag_traj(traj, [0,1]) for traj in innerTrajs]
            )
            trajsForClass = replace_rewards(trajsForClass, reward=lambda r: 1.)

            classifier.load_params(classOpt.get_value())
            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)

            if plot is not None:
                plot(trajsForClass, classifier)

            curiosityTrajs = replace_rewards(
                for_classifier(innerTrajs),
                model=classifier,
                reward=lambda o: o[1],
            )
            return innerTrajs, curiosityTrajs

        self.trajectories = trajectories
        self.render = inner.render
        self.reset_classifier = reset_classifier

from models.BaseWrapper import BaseWrapper

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]

def change_obs_space(trajs, changer=lambda x:x):
    return [[(changer(o, r),a,r) for o, a, r in t] for t in trajs]

def interesting_part(obs, r):
    result = (
            obs[0],
            obs[1],
            obs[2],
            obs[3],
												r
            )
    return result

STATE_SIZE = 24
ACTION_SIZE = 4
MAX_STEPS = 500

def walker():
    walker = Input(STATE_SIZE)
    walker = Affine(walker, 64)
    walker = LReLU(walker)
    walker = Affine(walker, ACTION_SIZE)
    walker = Tanh(walker)
    return walker

def run():
    classifier = Input(5)
    classifier = Affine(classifier, 32)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    agent = walker()
    agent.load_params(np.random.randn(agent.n_params)*1.5)
    agents = [walker() for _ in range(AGENT_VARIATIONS-1)]

    MAX_TRAIN_TIME = 400
    trainTimeLeft = MAX_TRAIN_TIME
    curAgentId = -1
    def plot_tagged_trajs(trajs, classifier):
        nonlocal trainTimeLeft, curAgentId
        COLORS = ["blue", "red"]
        coords = np.mgrid[0:11,0:11,0:11,0:11,0:11].reshape(5, -1).T * [0.4, 0.04, 0.25, 0.25, 0.2] - [2.0, 0.2, 1.25, 1.25, 1.0]
        classifierResults = classifier.outputs(coords)[:,1].reshape(11,11,11,11,11)
        classifierResults = np.mean(classifierResults, axis=(0,1,4)).T[::-1,:]
        plt.clf()
        plt.suptitle("Episode %d of agent %d"%(MAX_TRAIN_TIME-trainTimeLeft, curAgentId))
        for traj in trajs:
            tag = traj[0][1]
            xs, ys = [], []
            for state, _, _ in traj:
                x = state[2]
                y = state[3]
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, color=COLORS[np.argmax(tag)], alpha=0.1)
        plt.imshow(classifierResults, zorder=0, aspect="auto", vmin=0.0, vmax=1.0, cmap="gray", interpolation="bicubic", extent=[-1.25, 1.25, -1.25, 1.25])
        plt.gcf().set_size_inches(10, 8)
        plt.gcf().savefig(
            "__step_a%03d_t%03d.png" %
                (curAgentId, MAX_TRAIN_TIME-trainTimeLeft),
            dpi=100
        )


    world = Gym("BipedalWalker-v2", max_steps=MAX_STEPS)
    world = ActionNoise(world, stddev=0.2)
    world = Curiosity(
                world,
                classifier=classifier,
                for_classifier=lambda ts: change_obs_space(
                    ts,
                    changer=interesting_part
                    ),
                plot=plot_tagged_trajs
            )

    agentOpt = None

    def save_agent():
        np.save(
            "__ranger_a%03d_t%03d.npy" %
                (curAgentId, MAX_TRAIN_TIME-trainTimeLeft),
            agentOpt.get_value()
        )
    def reset_agent():
        nonlocal agentOpt, trainTimeLeft, curAgentId
        print("Resetting agent %d."%curAgentId)
        agentOpt = Adam(
            np.random.randn(agent.n_params)*1.5,
            lr=0.05,
            memory=0.9,
        )
        trainTimeLeft = MAX_TRAIN_TIME
        curAgentId += 1
    reset_agent()
    while True:
        parameters = agentOpt.get_value()

        agent.load_params(parameters)
        for i in range(AGENT_VARIATIONS-1):
            agents[i].load_params(parameters + np.random.randn(*parameters.shape)*0.9)
        realTrajs, curiosityTrajs = world.trajectories(agents+[agent], TRAJS_PER_VARIATION)

        print_reward(realTrajs, max_value=100.0, episode=np.sum, label="Real reward:    ")
        print_reward(curiosityTrajs, max_value=1.0, episode=np.max, label="Curiosity reward: ")
        if trainTimeLeft % 20 == 0:
            save_agent()

        if trainTimeLeft < 0:
            print("Timeout.")
            save_agent()
            trainTimeLeft = MAX_TRAIN_TIME
            reset_agent()
            continue

        realTrajs = discount(realTrajs, horizon=200)
        realTrajs = normalize(realTrajs)
        curiosityTrajs = replace_rewards(curiosityTrajs, episode=np.max)
        realWeight = 1.
        curiosityWeight = 1. - realWeight
        trajs = combine_rewards([realTrajs, curiosityTrajs], [realWeight, curiosityWeight])
        trajs = normalize(trajs)
        grad = policy_gradient(trajs, policy=agent)
        agentOpt.apply_gradient(grad)

        trainTimeLeft -= 1

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        world = Gym("BipedalWalker-v2", max_steps=MAX_STEPS)
        agent = walker()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        run()
