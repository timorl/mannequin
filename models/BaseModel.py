
class BaseModel(object):
    # Note: models are NOT assumed to be thread-safe
    # (calls can only be made from one thread at a time)

    def get_n_inputs(self):
        raise NotImplementedError

    def get_n_outputs(self):
        raise NotImplementedError

    def get_n_states(self):
        raise NotImplementedError

    def get_n_params(self):
        raise NotImplementedError

    def load_params(self, params):
        raise NotImplementedError

    def outputs(self, inputs):
        raise NotImplementedError

    def param_gradient_sum(self, inputs, output_gradients):
        raise NotImplementedError

    def input_gradients(self, inputs, output_gradients):
        raise NotImplementedError

    # For convenience only
    def __getattr__(self, name):
        if name == "n_inputs": return self.get_n_inputs()
        if name == "n_outputs": return self.get_n_outputs()
        if name == "n_states": return self.get_n_states()
        if name == "n_params": return self.get_n_params()
        raise AttributeError
