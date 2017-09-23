
from . import BaseWorld

class Bytes(BaseWorld):
    def __init__(self, *sequences, max_steps=None):
        import numpy as np

        sequences = [bytes(s) for s in sequences]
        assert len(sequences) >= 1

        one_hot = np.eye(257, 256)
        one_hot.setflags(write=False)

        rng = np.random.RandomState()

        def encode(seq):
            traj = [(one_hot[256], one_hot[seq[0]], 1.0)]
            for cur_b, next_b in zip(seq, seq[1:]):
                traj.append((one_hot[cur_b], one_hot[next_b], 1.0))
            return traj

        def trajectories(agent, n):
            assert agent == None
            assert n >= 1

            trajs = []

            for i in rng.randint(len(sequences), size=n):
                seq = sequences[i]

                if max_steps is not None:
                    assert isinstance(max_steps, int)
                    assert max_steps >= 1

                    if len(seq) > max_steps:
                        # Start at a random point
                        start = rng.randint(len(seq) - max_steps + 1)
                        seq = seq[start:start+max_steps]

                trajs.append(encode(seq))

            return trajs

        self.trajectories = trajectories
