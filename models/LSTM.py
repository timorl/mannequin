
from . import BaseTFModel
from ._verify_shapes import verify_shapes

@verify_shapes
class LSTM(BaseTFModel):
    def __init__(self, tf_model):
        import tensorflow as tf
        import tensorflow.contrib.rnn as rnn
        import numpy as np

        def _build_output_tensor(state_in, state_out):
            x = tf_model._build_output_tensor(state_in, state_out)

            # Make sure the first dimension is batch
            x_shape = x.shape.as_list()
            assert x_shape[0] is None

            # Flatten
            in_size = np.prod(x_shape[1:])
            x = tf.reshape(x, [-1, in_size])

            # Prepare input states
            c_state = tf.placeholder(tf.float32, [None, in_size])
            m_state = tf.placeholder(tf.float32, [None, in_size])
            state_in(c_state)
            state_in(m_state)

            # Compute LSTM step
            with tf.variable_scope(None, default_name="lstm"):
                lstm = rnn.BasicLSTMCell(in_size)
                x, (c_state, m_state) = lstm(x, (c_state, m_state))
                state_out(c_state)
                state_out(m_state)

            return x

        self._build_output_tensor = _build_output_tensor
