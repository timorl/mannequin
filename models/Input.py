
from . import BaseTFModel
from ._verify_shapes import verify_shapes

@verify_shapes
class Input(BaseTFModel):
    def __init__(self, *shape):
        import tensorflow as tf

        shape = tuple(max(1, int(d)) for d in shape)

        def _build_output_tensor(state_in, state_out):
            return tf.placeholder(
                tf.float32,
                (None,) + shape,
                name="inputs"
            )

        self._build_output_tensor = _build_output_tensor
