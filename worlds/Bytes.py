
from . import BaseWorld

class Bytes(BaseWorld):
    def __init__(self, *sequences, max_steps=1024):
        import numpy as np

        max_steps = int(max_steps)
        assert max_steps >= 1

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
                if len(seq) > max_steps:
                    # Start at a random point
                    start = rng.randint(len(seq) - max_steps + 1)
                    seq = seq[start:start+max_steps]

                trajs.append(encode(seq))

            return trajs

        def render(agent):
            obs = one_hot[256]
            output = []

            for _ in range(max_steps):
                (act,) = agent.outputs([obs])

                if len(act) > 256:
                    # Preserve model state
                    state = act[256:]
                    act = act[:256]
                else:
                    state = None

                act = np.reshape(act, 256)
                b = np.random.choice(256, p=act)
                output.append(b)

                obs = one_hot[b]
                if state is not None:
                    obs = np.concatenate((obs, state))

            print(repr(bytes(output)))

        self.trajectories = trajectories
        self.render = render
