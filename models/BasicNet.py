
from .BaseTFModel import BaseTFModel

class BasicNet(BaseTFModel):
    def __init__(self, layers):
        import numpy as np
        import tensorflow as tf

        assert len(layers) >= 2
        assert isinstance(layers[0], int)
        assert isinstance(layers[-1], int)

        self.get_input_shape = lambda: (layers[0],)
        self.get_output_shape = lambda: (layers[-1],)
        self.get_reward_shape = lambda: (layers[-1],)

        def affine(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            w = tf.Variable(tf.zeros([in_dim, out_dim]))
            b = tf.Variable(tf.zeros([out_dim]))
            return (tf.matmul(x, w) + b) / np.sqrt(in_dim + 1)

        def tf_init(graph, sess):
            inputs_in = tf.placeholder(tf.float32, [None, layers[0]])

            # Build layers
            outputs = inputs_in
            for l in layers[1:]:
                if isinstance(l, int):
                    outputs = affine(outputs, l)
                elif l == "relu":
                    outputs = tf.nn.relu(outputs)
                else:
                    raise ValueError("Unknown layer type: %s" % l)

            # Backpropagation
            params = graph.get_collection("variables")
            rewards_in = tf.placeholder(tf.float32, [None, layers[-1]])
            intermediate = tf.reduce_sum(
                tf.multiply(rewards_in, outputs)
            )
            grad_list = tf.gradients(intermediate, params)
            grad_list = [tf.reshape(g, [-1]) for g in grad_list]
            param_grad = tf.concat(grad_list, axis=0)

            def param_gradient(trajectories):
                all_inputs = []
                all_rewards = []
                for t in trajectories:
                    for i, _, r in t:
                        all_inputs.append(i)
                        all_rewards.append(r / len(t))

                grad = sess.run(
                    param_grad,
                    feed_dict={
                        inputs_in: all_inputs,
                        rewards_in: all_rewards
                    }
                )

                assert grad.shape == (self.n_params,)
                return grad / len(trajectories)

            def step(states, inputs):
                return states, sess.run(
                    outputs,
                    feed_dict={inputs_in: inputs}
                )

            self.param_gradient = param_gradient
            self.step = step
            return params

        super().__init__(tf_init)
