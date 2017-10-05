
from . import BaseTFModel
from ._verify_shapes import verify_shapes

@verify_shapes
class Affine(BaseTFModel):
    def __init__(self, inner, out_size):
        import tensorflow as tf
        import numpy as np

        out_size = int(out_size)

        def _build_output_tensor(state_in, state_out):
            x = inner._build_output_tensor(state_in, state_out)

            # Make sure the first dimension is batch
            x_shape = x.shape.as_list()
            assert x_shape[0] is None

            # Flatten
            in_size = np.prod(x_shape[1:])
            x = tf.reshape(x, (-1, in_size))

            # Affine transform
            w = tf.Variable(tf.zeros([in_size, out_size]))
            b = tf.Variable(tf.zeros([out_size]))
            x = (tf.matmul(x, w) + b) / np.sqrt(in_size + 1)

            return x

        self._build_output_tensor = _build_output_tensor
