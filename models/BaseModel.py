
class BaseModel(object):
    # Note: models are NOT assumed to be thread-safe
    # (calls can only be made from one thread at a time)

    def get_n_inputs(self):
        raise NotImplementedError

    def get_n_outputs(self):
        raise NotImplementedError

    def get_n_params(self):
        raise NotImplementedError

    def load_params(self, params):
        raise NotImplementedError

    def outputs(self, inputs):
        raise NotImplementedError

    def param_gradient(self, inputs, output_gradients):
        raise NotImplementedError

    def input_gradients(self, inputs, output_gradients):
        raise NotImplementedError

    # For convenience only
    def __getattr__(self, name):
        get = self.__class__.__getattribute__
        try:
            return get(self, "get_" + name)()
        except AttributeError:
            pass
        return get(self, name)
