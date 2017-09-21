#!/usr/bin/python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy, BaseWorld
from models import Input, Layer, Softmax, Constant
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, retrace, get_reward, process_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

class Curiosity(BaseWorld.BaseWorld):
    def __init__(self, inner, *, classifier, initial_agent, history_length, plot=False):
        history = inner.trajectories(initial_agent, history_length)
        classOpt = Adam(
            np.random.randn(classifier.n_params) * 1.,
            lr=0.5,
            memory=0.9,
        )

        def tag_traj(traj, tag):
            return [(t[0], tag, 1.) for t in traj]

        def plot_tagged_trajs(trajs):
            COLORS = ["blue", "red", "green"]
            plt.ion()
            plt.clf()
            plt.grid()
            plt.gcf().axes[0].set_ylim([-0.075,0.075])
            plt.gcf().axes[0].set_xlim([-1.25,0.5])
            for traj in trajs:
                tag = traj[0][1]
                xs, ys = [], []
                for (x,y), _, _ in traj:
                    xs.append(x)
                    ys.append(y)
                plt.plot(xs, ys, color=COLORS[np.argmax(tag)])
            plt.pause(0.01)

        def trained_agent(agent):
            nonlocal history
            history += inner.trajectories(agent, history_length)
            classOpt = Adam(
                np.random.randn(classifier.n_params) * 1.,
                lr=0.5,
                memory=0.9,
            )

        def trajectories(agent, n):
            nonlocal history, classOpt
            classifier.load_params(classOpt.get_value())
            historyIdx = np.random.choice(len(history), size=n)
            innerTrajs = [history[i] for i in historyIdx]
            innerTrajs += inner.trajectories(agent, n)
            trajsForClass = [tag_traj(traj, [1,0]) for traj in innerTrajs[:n]]
            trajsForClass += [tag_traj(traj, [0,1]) for traj in innerTrajs[n:]]
            if plot:
                plot_tagged_trajs(trajsForClass)
            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)

            innerTrajs = innerTrajs[n:]
            preds = retrace(innerTrajs, model=classifier)
            curiosityTrajs = [[(o, a, p) for (o, a, _), (_, (_, p)) in zip(innerTraj, pred)] for innerTraj, pred in zip(innerTrajs, preds)]
            return innerTrajs, curiosityTrajs

        self.trajectories = trajectories
        self.render = inner.render
        self.trained_agent = trained_agent

from models.BaseWrapper import BaseWrapper

class SlightlyGaussianAgent(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        rng = np.random.RandomState()

        def smear(v):
            assert v.shape == (1,)
            v[0] = v[0]+np.random.randn()*0.1
            return v

        def step(states, inputs):
            states, outputs = inner.step(states, inputs)
            outputs = np.array(outputs)

            for i in range(len(outputs)):
                outputs[i] = smear(outputs[i])

            return states, outputs

        super().__init__(inner, step=step)

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]


def carr():
    carr = Input(2)
    carr = Layer(carr, 32, "lrelu")
    carr = Layer(carr, 1)
    return SlightlyGaussianAgent(carr)

def run():
    classifier = Input(2)
    classifier = Layer(classifier, 16, "lrelu")
    classifier = Layer(classifier, 2)
    classifier = Softmax(classifier)

    curCarr = carr()
    curCarr.load_params(np.random.randn(curCarr.n_params))

    world = Gym("MountainCarContinuous-v0", max_steps=500)
    world = Curiosity(
                world,
                classifier=classifier,
                initial_agent=curCarr,
                history_length=800,
                plot=True
            )

    def train_one(carrOpt):
        if carrOpt == None:
            carrOpt = Adam(
                np.random.randn(curCarr.n_params),
                lr=0.10,
                memory=0.5,
            )
        nextBreak = 5
        for i in range(250):
            curCarr.load_params(carrOpt.get_value())

            realTrajs, curiosityTrajs = world.trajectories(curCarr, 50)
            curScore = get_reward(realTrajs, episode=np.sum, episodes=np.mean)/90.
            print_reward(realTrajs, max_value=90.0, episode=np.sum, label="Real reward:      ")
            print_reward(curiosityTrajs, max_value=1.0, episode=np.max, label="Curiosity reward: ")
            curCuriosity = get_reward(curiosityTrajs, episode=np.max, episodes=np.mean)
            if curCuriosity > 0.98:
                if nextBreak == 0:
                    break
                else:
                    nextBreak -= 1
            else:
                nextBreak = np.min([nextBreak+1, 5])

            realTrajs = process_rewards(realTrajs, episode=np.sum)
            realTrajs = normalize(realTrajs)
            curiosityTrajs = process_rewards(curiosityTrajs, episode=np.max)
            #this is stupid, we should care more(?) if the costs are to high
            realWeight = 0.001 + np.max([np.min([curScore, 0.2]), 0.])*0.998/0.2
            curiosityWeight = 1. - realWeight
            print('RWeight: %f, CWeight: %f'%(realWeight, curiosityWeight))
            trajs = combine_rewards([realTrajs, curiosityTrajs], [realWeight, curiosityWeight])
            trajs = normalize(trajs)
            grad = policy_gradient(trajs, policy=curCarr)
            carrOpt.apply_gradient(grad)
            if i % 10 == 0:
                print("%d episodes in."%i)
        world.trained_agent(curCarr)
        world.render(curCarr)
        if curScore > 0.01:
            return carrOpt
        else:
            return None
    theCarOpt = None
    for i in range(50):
        print("Teaching agent %d."%i)
        theCarOpt = train_one(theCarOpt)

if __name__ == "__main__":
    run()
