
from . import BaseWorld

class Mnist(BaseWorld):
    data = None

    def __init__(self, *, test=False):
        import numpy as np

        if Mnist.data == None:
            import tensorflow.examples.tutorials.mnist as tf_mnist
            Mnist.data = tf_mnist.input_data.read_data_sets(
                "/tmp/mnist-download",
                validation_size=0,
                one_hot=True
            )

        data = Mnist.data.test if test else Mnist.data.train

        def trajectories(agent, n):
            assert agent == None
            assert n >= 1

            inputs, labels = data.next_batch(n)
            return [
                [(i.reshape(28, 28), l, 1.0)]
                for i, l in zip(inputs, labels)
            ]

        self.trajectories = trajectories
