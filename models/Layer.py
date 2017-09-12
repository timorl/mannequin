
from .BaseTFModel import BaseTFModel

class Layer(BaseTFModel):
    def __init__(self, tf_model, out_size, activation=None):
        import tensorflow as tf
        import numpy as np

        out_size = int(out_size)

        def build_output_tensor():
            x = tf_model.build_output_tensor()

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

            # Apply activation to the output
            if activation is None:
                pass
            elif activation == "relu":
                x = tf.nn.relu(x)
            elif activation == "lrelu":
                x = tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)
            elif activation == "tanh":
                x = tf.nn.tanh(x)
            else:
                raise ValueError("Unknown activation type: %s" % l)

            return x

        self.build_output_tensor = build_output_tensor
