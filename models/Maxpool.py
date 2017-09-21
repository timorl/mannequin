
from . import BaseTFModel

class Maxpool(BaseTFModel):
    def __init__(self, tf_model, *, size, padding="SAME"):
        import tensorflow as tf
        import numpy as np

        size = int(size)
        padding = str(padding)

        def build_output_tensor():
            x = tf_model.build_output_tensor()

            # Make sure the first dimension is batch
            x_shape = x.shape.as_list()
            assert x_shape[0] is None
            x_shape[0] = -1

            if len(x_shape) == 3:
                # Input has one channel
                x_shape += [1]
                x = tf.reshape(x, [-1] + x_shape[1:])
            elif len(x_shape) != 4:
                raise ValueError("Maxpool: Invalid input shape")

            return tf.nn.max_pool(
                x,
                ksize=[1, size, size, 1],
                strides=[1, size, size, 1],
                padding=padding
            )

        self.build_output_tensor = build_output_tensor
