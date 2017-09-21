
from . import BaseModel

class BaseTFModel(BaseModel):
    def build_output_tensor(self):
        raise NotImplementedError

    def __getattribute__(self, name):
        lazy_attributes = (
            "get_input_shape",
            "get_output_shape",
            "get_n_params",
            "load_params",
            "param_gradient",
            "step",
        )

        try:
            if name in lazy_attributes:
                object.__getattribute__(self, "is_initialized")
        except AttributeError:
            self.is_initialized = True

            # Lazy initialization
            import tensorflow as tf
            graph = tf.Graph()
            sess = tf.Session(graph=graph)
            with graph.as_default():
                with sess:
                    self.initialize(graph, sess)
            tf.reset_default_graph()

        return object.__getattribute__(self, name)

    def initialize(self, graph, sess):
        import sys
        import numpy as np
        import tensorflow as tf

        # Build graph
        outputs = self.build_output_tensor()
        inputs = graph.get_tensor_by_name("inputs:0")
        params = graph.get_collection("variables")

        # Count parameters
        n_params = 0
        for p in params:
            n_params += np.prod(p.shape.as_list())
        if n_params < 1:
            raise ValueError("Cannot build a model with 0 parameters")
        sys.stderr.write("TensorFlow model: %d parameters\n" % n_params)

        # Expose shapes in the public interface
        input_shape = tuple(inputs.shape.as_list()[1:])
        output_shape = tuple(outputs.shape.as_list()[1:])
        self.get_input_shape = lambda: input_shape
        self.get_output_shape = lambda: output_shape
        self.get_n_params = lambda: n_params

        # Create ops to load all parameters from a single array
        params_in = tf.placeholder(tf.float32, n_params)
        load_ops = []
        pos = -1, 0
        for p in params:
            pos = pos[1], pos[1] + np.prod(p.shape.as_list())
            load_ops.append(p.assign(
                tf.reshape(params_in[pos[0]:pos[1]], p.shape)
            ))
        assert pos[1] == n_params

        # Backpropagation
        out_grad_in = tf.placeholder(tf.float32, outputs.shape)
        intermediate = tf.reduce_sum(tf.reduce_mean(
            tf.multiply(out_grad_in, outputs),
            axis=0 # (batch average)
        ))
        grad_list = tf.gradients(intermediate, params)
        grad_list = [tf.reshape(g, [-1]) for g in grad_list]
        param_grad = tf.concat(grad_list, axis=0)

        def load_params(feed_params):
            sess.run(
                load_ops,
                feed_dict={params_in: feed_params}
            )

        def param_gradient(states, feed_inputs, output_gradients):
            return sess.run(
                param_grad,
                feed_dict={
                    inputs: feed_inputs,
                    out_grad_in: output_gradients
                }
            )

        def step(states, feed_inputs):
            return states, sess.run(
                outputs,
                feed_dict={inputs: feed_inputs}
            )

        self.load_params = load_params
        self.param_gradient = param_gradient
        self.step = step
