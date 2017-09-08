
from . import Model

class BaseTFModel(Model):
    def __init__(self, tf_init):
        import os
        import sys
        import numpy as np
        import tensorflow as tf

        # Build graph and get a list parameter tensors
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            with sess:
                params = [p for p in tf_init(graph, sess)]

        # Count parameters
        n_params = 0
        for p in params:
            n_params += np.prod(p.shape.as_list())
        assert n_params >= 1

        # Create ops to load all parameters from a single array
        with graph.as_default():
            params_in = tf.placeholder(tf.float32, n_params)
            load_ops = []
            pos = -1, 0
            for p in params:
                pos = pos[1], pos[1] + np.prod(p.shape.as_list())
                value = tf.reshape(params_in[pos[0]:pos[1]], p.shape)
                load_ops.append(p.assign(value))

        # Expose interface
        self.get_n_params = lambda: n_params
        self.load_params = (
            lambda p: sess.run(load_ops, feed_dict={params_in: p})
        )

        if "DEBUG" in os.environ:
            sys.stderr.write("Model %s has %d parameters\n"
                % (self, n_params))

        tf.reset_default_graph()
