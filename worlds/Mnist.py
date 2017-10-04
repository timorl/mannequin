
from . import BaseTable

class Mnist(BaseTable):
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
        super().__init__(data.images, data.labels)
