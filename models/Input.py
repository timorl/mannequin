
from .BaseTFModel import BaseTFModel

class Input(BaseTFModel):
    def __init__(self, *shape):
        import tensorflow as tf

        shape = tuple(max(1, int(d)) for d in shape)

        def build_output_tensor():
            return tf.placeholder(
                tf.float32,
                (None,) + shape,
                name="inputs"
            )

        self.build_output_tensor = build_output_tensor
