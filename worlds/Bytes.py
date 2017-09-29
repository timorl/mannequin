
from . import BaseWorld

class Bytes(BaseWorld):
    def __init__(self, *sequences, max_steps=1024, charset=None):
        import numpy as np

        sequences = [bytes(s) for s in sequences]
        assert len(sequences) >= 1

        max_steps = int(max_steps)
        assert max_steps >= 1

        if charset is None:
            charset = range(256)
        charset = sorted(list(set(charset)))
        assert len(charset) >= 1
        assert charset[0] >= 0 and charset[-1] <= 255

        n_inputs = len(charset)
        one_hot = np.eye(n_inputs)
        one_hot.setflags(write=False)
        zeros = np.zeros(n_inputs)
        zeros.setflags(write=False)

        encoding = [zeros] * 256
        for i, c in enumerate(charset):
            encoding[c] = one_hot[i]

        rng = np.random.RandomState()

        def encode(seq):
            traj = [(zeros, encoding[seq[0]], 1.0)]
            for a, b in zip(seq, seq[1:]):
                traj.append((encoding[a], encoding[b], 1.0))
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
            obs = np.zeros(n_inputs + agent.n_states)
            output = []

            for _ in range(max_steps):
                (act,) = agent.outputs([obs])
                act = np.reshape(act, -1)
                state = act[n_inputs:]
                act = act[:n_inputs]

                b = np.random.choice(n_inputs, p=act)
                output.append(charset[b])

                obs = one_hot[b]
                obs = np.concatenate((obs, state))

            print(repr(bytes(output)))

        self.trajectories = trajectories
        self.render = render
