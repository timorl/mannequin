#!/usr/bin/python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Accuracy, Normalized, Future, PrintReward
from Friedrich import Friedrich
from Curiosity import Curiosity
from models import Input, Layer, Softmax, Constant
from optimizers import Adam
from execute import policy_gradient

if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

def run():
    classifier = Input(2)
    classifier = Layer(classifier, 16, "lrelu")
    classifier = Layer(classifier, 2)
    classifier = Softmax(classifier)

    gaussCenterer = Constant(2)

    world = Friedrich(gaussCenterer)
    curious = Curiosity(world, classifier)
    curious = Normalized(curious)

    testWorld = PrintReward(Accuracy(world), max_value=100.)

    classOpt = Adam(
        np.random.randn(classifier.n_params) * 0.1,
        lr=0.5,
        mean_decay=0.99
    )
    gaussOpt = Adam(
        [0,0],
        lr=0.05,
        mean_decay=0.2
    )

    plt.ion()
    plt.scatter([0],[0])
    plt.grid()
    values = []
    for _ in range(100):
        classifier.load_params(classOpt.get_value())
        gaussCenterer.load_params(gaussOpt.get_value())
        trajs = world.trajectories(None, 100)
        testWorld.trajectories(classifier, 1000)
        grad = policy_gradient(classifier, trajs)
        classOpt.apply_gradient(grad)
        trajs2 = curious.trajectories(None, 10)
        grad2 = policy_gradient(gaussCenterer, trajs2)
        values.append(gaussOpt.get_value())
        plt.plot(*zip(*values))
        plt.pause(0.1)
        gaussOpt.apply_gradient(grad2)

if __name__ == "__main__":
    run()
