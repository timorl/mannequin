
from . import World

class Mnist(World):
    data = None

    def __init__(self, test=False):
        import numpy as np

        if Mnist.data == None:
            import tensorflow.examples.tutorials.mnist as tf_mnist
            Mnist.data = tf_mnist.input_data.read_data_sets(
                "/tmp/mnist-download",
                validation_size=0,
                one_hot=True
            )

        data = Mnist.data.test if test else Mnist.data.train
        rng = np.random.RandomState()

        def trajectories(agent, n):
            assert n >= 1
            inputs, labels = data.next_batch(n)

            # Get predictions from the agent
            _, outputs = agent.step([None] * len(inputs), inputs)

            # Ensure shapes are correct
            labels = np.asarray(labels)
            outputs = np.asarray(outputs)
            assert labels.shape == outputs.shape

            # Gradient (label - prediction) is correct for
            # MSE and also for softmax + cross entropy (!)
            return [
                [(i, o, l - o)]
                for i, o, l in zip(inputs, outputs, labels)
            ]

        self.trajectories = trajectories
