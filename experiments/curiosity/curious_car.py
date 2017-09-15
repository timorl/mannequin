#!/usr/bin/python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy
from models import Input, Layer, Softmax, Constant
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, retrace, episode_accumulate_reward

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

def convert_traj(traj, pred, class_id):
    result = [(o, a, p[class_id]) for (o, a, _), (_, p) in zip(traj,pred)]
    return result

def learn_from_classifier(classifier, trajs, class_id):
    preds = retrace(trajs, model=classifier)
    return [convert_traj(traj, pred, class_id) for traj, pred in zip(trajs, preds)]

def carr():
    carr = Input(2)
    carr = Layer(carr, 32, "lrelu")
    carr = Layer(carr, 3)
    return Softmax(carr)

def tag_traj(traj, tag):
    return [(t[0], tag, 1.) for t in traj]

def plot_tagged_trajs(trajs):
    COLORS = ["blue", "red", "green"]
    plt.ion()
    plt.clf()
    plt.grid()
    plt.gcf().axes[0].set_ylim([-0.2,0.2])
    plt.gcf().axes[0].set_xlim([-1.5,1.5])
    for traj in trajs:
        tag = traj[0][1]
        xs, ys = [], []
        for (x,y), _, _ in traj:
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys, color=COLORS[np.argmax(tag)])
    plt.pause(0.01)

def run():
    classifier = Input(2)
    classifier = Layer(classifier, 16, "lrelu")
    classifier = Layer(classifier, 2)
    classifier = Softmax(classifier)

    world = Gym("MountainCar-v0")
    world = StochasticPolicy(world)

    carrs = [carr()]
    carrs[0].load_params(np.random.randn(carrs[0].n_params))

    def train_one():
        curCarr = carr()
        carrs.append(curCarr)
        classOpt = Adam(
            np.random.randn(classifier.n_params) * 1.,
            lr=0.5,
            memory=0.9,
        )
        carrOpt = Adam(
            np.random.randn(curCarr.n_params),
            lr=0.10,
            memory=0.5,
        )
        curAccuracy = 0.
        for i in range(200):
            classifier.load_params(classOpt.get_value())
            curCarr.load_params(carrOpt.get_value())

            trajs = []
            for jimmy in np.random.choice(carrs[:-1], size=50):
                trajs += world.trajectories(jimmy, 1)
            trajs += world.trajectories(curCarr, 50)
            trajsForClass = [tag_traj(traj, [1,0]) for traj in trajs[:50]]
            trajsForClass += [tag_traj(traj, [0,1]) for traj in trajs[50:]]
            plot_tagged_trajs(trajsForClass)
            accTrajs = accuracy(trajsForClass, model=classifier)
            accTrajs = episode_accumulate_reward(accTrajs, np.mean)
            print_reward(accTrajs, max_value=1.0, reward_accumulator=np.mean)
            accs = [traj[0][2] for traj in accTrajs]
            curAccuracy = np.mean(accs)
            if curAccuracy > 1.-i/400:
                break

            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)
            trajs2 = learn_from_classifier(classifier, trajs[50:], 1)
            trajs2 = episode_accumulate_reward(trajs2, accumulator=np.max)
            trajs2 = normalize(trajs2)
            grad2 = policy_gradient(trajs2, policy=curCarr)
            carrOpt.apply_gradient(grad2)
            if i % 10 == 0:
                print("%d episodes in."%i)
        world.render(curCarr)
    for i in range(10):
        print("Teaching agent %d."%i)
        train_one()

if __name__ == "__main__":
    run()
