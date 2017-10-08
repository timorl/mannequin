#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy, BaseWorld, ActionNoise, Cache
from models import Input, Affine, LReLU, Softmax, Constant
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, get_rewards, replace_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

class Curiosity(BaseWorld):
    def __init__(self, inner, *, classifier, history_length, plot=False):
        history = Cache()
        classOpt = None

        def tag_traj(traj, tag):
            return [(t[0], tag, 1.) for t in traj]

        def plot_tagged_trajs(trajs):
            COLORS = ["blue", "red"]
            plt.ion()
            plt.clf()
            plt.grid()
            plt.gcf().axes[0].set_ylim([-1.25,1.25])
            plt.gcf().axes[0].set_xlim([-1.25,1.25])
            for traj in trajs:
                tag = traj[0][1]
                xs, ys = [], []
                for state, _, _ in traj:
                    x = state[2]
                    y = state[3]
                    xs.append(x)
                    ys.append(y)
                plt.plot(xs, ys, color=COLORS[np.argmax(tag)], alpha=0.1)
            plt.pause(0.01)

        def remember(agent):
            nonlocal history, classOpt
            history.add_trajectory(
                *inner.trajectories(agent, history_length)
            )
            classOpt = Adam(
                np.random.randn(classifier.n_params) * 1.,
                lr=0.08,
                memory=0.9,
            )

        def trajectories(agent, n):
            if classOpt is None:
                remember(agent)

            oldTrajs = history.trajectories(None, n)
            innerTrajs = inner.trajectories(agent, n)

            trajsForClass = (
                [tag_traj(traj, [1,0]) for traj in oldTrajs]
                + [tag_traj(traj, [0,1]) for traj in innerTrajs]
            )

            if plot:
                plot_tagged_trajs(trajsForClass)

            classifier.load_params(classOpt.get_value())
            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)

            curiosityTrajs = replace_rewards(
                innerTrajs,
                model=classifier,
                reward=lambda o: o[1]
            )
            return innerTrajs, curiosityTrajs

        self.trajectories = trajectories
        self.render = inner.render
        self.remember = remember

from models.BaseWrapper import BaseWrapper

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]

STATE_SIZE = 24
ACTION_SIZE = 4

def walker():
    walker = Input(STATE_SIZE)
    walker = Affine(walker, 64)
    walker = LReLU(walker)
    walker = Affine(walker, ACTION_SIZE)
    return walker

def run():
    classifier = Input(STATE_SIZE)
    classifier = Affine(classifier, 32)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    agent = walker()
    agent.load_params(np.random.randn(agent.n_params))

    world = Gym("BipedalWalker-v2", max_steps=800)
    world = ActionNoise(world, stddev=0.05)
    world = Curiosity(
                world,
                classifier=classifier,
                history_length=100,
                plot=True
            )

    MAX_BOREDOM = 3
    boredom = MAX_BOREDOM

    MAX_TRAIN_TIME = 500
    agentOpt = None
    trainTimeLeft = None
    lastScores = None

    curAgentId = -1
    curMemoryId = 0
    def memorize():
        nonlocal boredom, curMemoryId
        print("Memorizing %d..."%curMemoryId)
        world.remember(agent)
        boredom = MAX_BOREDOM
        curMemoryId += 1
    def reset_agent():
        nonlocal agentOpt, trainTimeLeft, lastScores, curAgentId
        print("Resetting agent %d."%curAgentId)
        agentOpt = Adam(
            np.random.randn(agent.n_params),
            lr=0.03,
            memory=0.8,
        )
        trainTimeLeft = MAX_TRAIN_TIME
        lastScores = [-0.4]
        curAgentId += 1
    reset_agent()
    while True:
        agent.load_params(agentOpt.get_value())

        realTrajs, curiosityTrajs = world.trajectories(agent, 20)
        curScore = np.mean(get_rewards(realTrajs, episode=np.sum))/150.
        lastScores.append(curScore)
        lastScores = lastScores[-10:]
        scoreDev = np.std(lastScores)

        curCuriosity = np.mean(get_rewards(curiosityTrajs, episode=np.max))

        print_reward(realTrajs, max_value=150.0, episode=np.sum, label="Real reward:      ")
        print_reward(curiosityTrajs, max_value=1.0, episode=np.max, label="Curiosity reward: ")
        if curCuriosity > 0.95:
            if boredom == 0:
                memorize()
                if curMemoryId % 3 == 0:
                    world.render(agent)
            else:
                boredom -= 1
        else:
            boredom = np.min([boredom+1, MAX_BOREDOM])

        if scoreDev < 0.01 or trainTimeLeft < 0:
            print("Not really learning.")
            world.render(agent)
            if curScore < 0.01: 
                memorize()
                reset_agent()
                continue

        realTrajs = replace_rewards(realTrajs, episode=np.sum)
        realTrajs = normalize(realTrajs)
        curiosityTrajs = replace_rewards(curiosityTrajs, episode=np.max)
        realWeight = np.min([(scoreDev-0.01)*6., 0.95])
        curiosityWeight = 1. - realWeight
        print('RWeight: %f, CWeight: %f'%(realWeight, curiosityWeight))
        trajs = combine_rewards([realTrajs, curiosityTrajs], [realWeight, curiosityWeight])
        trajs = normalize(trajs)
        grad = policy_gradient(trajs, policy=agent)
        agentOpt.apply_gradient(grad)

        trainTimeLeft -= 1
        if trainTimeLeft % 10 == 0:
            print("%d episodes in."%(MAX_TRAIN_TIME-trainTimeLeft))

if __name__ == "__main__":
    run()
