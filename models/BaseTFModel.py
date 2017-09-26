
from . import BaseModel

class BaseTFModel(BaseModel):
    def _build_output_tensor(self):
        raise NotImplementedError

    def __getattribute__(self, name):
        try:
            if not name.startswith("_"):
                object.__getattribute__(self, "is_initialized")
        except AttributeError:
            # Lazy initialization
            self.is_initialized = True

            try:
                import tensorflow as tf
                graph = tf.Graph()
                sess = tf.Session(graph=graph)
                with graph.as_default():
                    with sess:
                        BaseTFModel.initialize(self, graph, sess)
                tf.reset_default_graph()
            except Exception:
                raise ValueError("Could not initialize model")

        return super().__getattribute__(name)

    def initialize(self, graph, sess):
        import sys
        import numpy as np
        import tensorflow as tf

        # Build graph
        outputs = self._build_output_tensor()
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
        n_inputs = np.prod(inputs.shape.as_list()[1:])
        n_outputs = np.prod(outputs.shape.as_list()[1:])
        self.get_n_inputs = lambda: n_inputs
        self.get_n_outputs = lambda: n_outputs
        self.get_n_states = lambda: 0
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

        # Backpropagation to parameters
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

        def get_outputs(feed_inputs):
            inps_shape = inputs.shape.as_list()
            inps_shape[0] = len(feed_inputs)
            feed_inputs = np.reshape(feed_inputs, inps_shape)

            outs = sess.run(
                outputs,
                feed_dict={inputs: feed_inputs}
            )

            return np.reshape(outs, (len(feed_inputs), n_outputs))

        def get_param_gradient(feed_inputs, output_gradients):
            inps_shape = inputs.shape.as_list()
            inps_shape[0] = len(feed_inputs)
            feed_inputs = np.reshape(feed_inputs, inps_shape)

            outs_shape = outputs.shape.as_list()
            outs_shape[0] = len(feed_inputs)
            output_gradients = np.reshape(output_gradients, outs_shape)

            grad = sess.run(
                param_grad,
                feed_dict={
                    inputs: feed_inputs,
                    out_grad_in: output_gradients
                }
            )
            return np.reshape(grad, (n_params,))

        self.load_params = load_params
        self.outputs = get_outputs
        self.param_gradient = get_param_gradient
