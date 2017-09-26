
from . import BaseTFModel
from ._verify_shapes import verify_shapes

@verify_shapes
class Conv2d(BaseTFModel):
    def __init__(self, tf_model, *,
            size, channels, stride=1, padding="SAME"):
        import tensorflow as tf
        import numpy as np

        size = int(size)
        channels = int(channels)
        stride = int(stride)
        padding = str(padding)

        def _build_output_tensor():
            x = tf_model._build_output_tensor()

            # Make sure the first dimension is batch
            x_shape = x.shape.as_list()
            assert x_shape[0] is None

            if len(x_shape) == 3:
                # Input has one channel
                x_shape += [1]
                x = tf.reshape(x, [-1] + x_shape[1:])
            elif len(x_shape) != 4:
                raise ValueError("Conv2d: Invalid input shape")

            w = tf.Variable(tf.zeros(
                [size, size, x_shape[-1], channels]
            ))

            x = tf.nn.conv2d(
                x, w,
                strides=[1, stride, stride, 1],
                padding=padding
            )

            b = tf.Variable(tf.zeros([channels]))

            return x + b

        self._build_output_tensor = _build_output_tensor
