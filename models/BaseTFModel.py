
from . import BaseModel

class BaseTFModel(BaseModel):
    def _build_output_tensor(self, state_in, state_out):
        raise NotImplementedError

    def __getattribute__(self, name):
        initialize = False
        try:
            if not name.startswith("_"):
                object.__getattribute__(self, "is_initialized")
        except AttributeError:
            # Lazy initialization
            self.is_initialized = True
            initialize = True

        if initialize:
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

        def total_size(*tensors):
            n = 0
            for t in tensors:
                shape = t.shape.as_list()
                if shape[0] == None: shape = shape[1:]
                n += np.prod(shape)
            return n

        # Build graph
        states = []
        states_out = []
        outputs = self._build_output_tensor(
            states.append,
            states_out.append
        )
        inputs = graph.get_tensor_by_name("inputs:0")
        params = graph.get_collection("variables")

        # Count parameters
        n_params = total_size(*params)
        n_states = total_size(*states)
        assert n_states == total_size(*states_out)
        n_inputs = total_size(inputs)
        n_outputs = total_size(outputs)
        if n_params < 1:
            raise ValueError("Cannot build a model with 0 parameters")
        sys.stderr.write("TensorFlow model: %d parameters\n" % n_params)

        # Expose shapes in the public interface
        self.get_n_inputs = lambda: n_inputs
        self.get_n_outputs = lambda: n_outputs
        self.get_n_states = lambda: n_states
        self.get_n_params = lambda: n_params

        # Prepare flattened output with states
        outputs = [outputs] + states_out
        del states_out
        outputs = [tf.reshape(o, [-1, total_size(o)]) for o in outputs]
        outputs = tf.concat(outputs, axis=1)

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

        def feed(feed_inputs, feed_dict={}):
            batch = len(feed_inputs)
            feed_inputs = np.reshape(feed_inputs, (batch, -1))

            # Regular inputs
            feed_dict[inputs] = feed_inputs[:,:n_inputs].reshape(
                [batch] + inputs.shape.as_list()[1:]
            )

            # States
            pos = n_inputs
            for s in states:
                size = total_size(s)
                feed_dict[s] = feed_inputs[:,pos:pos+size].reshape(
                    [batch] + s.shape.as_list()[1:]
                )
                pos += size
            assert pos == feed_inputs.shape[1]

            return feed_dict

        def get_outputs(feed_inputs):
            return sess.run(outputs, feed_dict=feed(feed_inputs))

        def get_param_gradient(feed_inputs, output_gradients):
            return sess.run(param_grad, feed_dict=feed(
                feed_inputs,
                {out_grad_in: output_gradients}
            ))

        self.load_params = load_params
        self.outputs = get_outputs
        self.param_gradient = get_param_gradient
