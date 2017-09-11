
from .BaseWrapper import BaseWrapper
from ._verify_shapes import verify_shapes

@verify_shapes
class OffPolicy(BaseWrapper):
    def __init__(self, inner):
        import numpy as np

        def param_gradient(trajs):
            states = [None] * len(trajs)
            step = 0

            while True:
                # Find trajectories that are still active
                pos = [n for n, t in enumerate(trajs) if len(t) > step]
                if len(pos) < 1:
                    break

                # Unpack a slice across all active trajectories
                inp, real_out, rew = zip(*[trajs[n][step] for n in pos])

                # Get current model predictions
                sta = [states[n] for n in pos]
                sta, model_out = inner.step(sta, inp)

                # Verify array shapes
                rew = np.asarray(rew).reshape(-1)
                real_out = np.asarray(real_out)
                model_out = np.asarray(model_out)
                assert rew.shape == (len(pos),)
                assert len(real_out) == len(rew)
                assert real_out.shape == model_out.shape

                # True off-policy version of REINFORCE (!!!)
                rew = np.multiply((real_out - model_out).T, rew.T).T

                for n, s, i, o, r in zip(pos, sta, inp, model_out, rew):
                    states[n] = s
                    trajs[n][step] = (i, o, r)

                step += 1

            return inner.param_gradient(trajs)

        super().__init__(
            inner,
            get_reward_shape=lambda: (1,),
            param_gradient=param_gradient
        )
