
from . import BaseWrapper
from ._verify_shapes import verify_shapes

@verify_shapes
class Memory(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        length = inner.inp_shape[0]
        assert length >= 2

        inp_shape = inner.inp_shape[1:]
        assert len(inp_shape) >= 1

        null_inp = np.zeros(*inp_shape)
        null_inp.setflags(write=False)

        def remember(sta, inp):
            if sta is None:
                return (None, [null_inp] * (length-1) + [inp])
            else:
                mem = sta[1]
                assert len(mem) == length
                return (sta[0], mem[1:] + [inp])

        def step(states, inputs):
            # Add current inputs to memory
            states = [remember(s, i) for s, i in zip(states, inputs)]

            # The inner agent sees multiple inputs at once
            inner_states, outputs = inner.step(*zip(*states))
            for i in range(len(states)):
                states[i] = (inner_states[i], states[1][1])

            return states, outputs

        def param_gradient(states, inputs, output_gradients):
            # Add current inputs to memory
            states = [remember(s, i) for s, i in zip(states, inputs)]

            return inner.param_gradient(*zip(*states), output_gradients)

        super().__init__(inner,
            step=step,
            param_gradient=param_gradient,
            inp_shape=inp_shape)
