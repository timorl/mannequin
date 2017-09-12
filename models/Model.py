
class Model(object):
    # Note: models are NOT assumed to be thread-safe
    # (calls can only be made from one thread at a time)

    def get_input_shape(self):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError

    def get_n_params(self):
        raise NotImplementedError

    def load_params(self, params):
        raise NotImplementedError

    def param_gradient(self, states, inputs, output_gradients):
        raise NotImplementedError

    def step(self, states, inputs): # -> (states, outputs)
        raise NotImplementedError

    # For convenience only
    def __getattr__(self, name):
        if name in ("inp_shape", "input_shape"):
            return self.get_input_shape()
        if name in ("out_shape", "output_shape"):
            return self.get_output_shape()
        if name in ("n_params"):
            return self.get_n_params()
        return object.__getattribute__(self, name)
