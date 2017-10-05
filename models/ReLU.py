
from . import BaseTFModel
from ._verify_shapes import verify_shapes

@verify_shapes
class ReLU(BaseTFModel):
    def __init__(self, inner):
        import tensorflow as tf

        def _build_output_tensor(state_in, state_out):
            x = inner._build_output_tensor(state_in, state_out)

            x = tf.nn.relu(x);
            return x

        self._build_output_tensor = _build_output_tensor
