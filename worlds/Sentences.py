
from . import BaseWorld

class Sentences(BaseWorld):
    def __init__(self, text):
        import numpy as np

        one_hot = np.eye(65, 64)
        one_hot.setflags(write=False)

        charset = (
            ' .!?,:;()-/"'
            + 'abcdefghijklmnopqrstuvwxyz'
            + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )
        assert len(charset) == 64
        encode, decode = encoding(charset)

        text = str(text)
        ends = [i+1 for i, c in enumerate(text) if c in ".!?"]
        sentences = [encode(text[i:j]) for i, j in zip([0]+ends, ends)]
        sentences = [s for s in sentences if len(s) >= 2]
        assert len(sentences) >= 1

        rng = np.random.RandomState()

        def experience(seq):
            traj = [(one_hot[64], one_hot[seq[0]], 1.0)]
            for cur_b, next_b in zip(seq, seq[1:]):
                traj.append((one_hot[cur_b], one_hot[next_b], 1.0))
            return traj

        def trajectories(agent, n):
            assert agent == None
            assert n >= 1

            return [
                experience(sentences[i])
                for i in rng.randint(len(sentences), size=n)
            ]

        def render(agent):
            sta = None
            obs = one_hot[64]
            output = []
            for _ in range(1000):
                (sta,), (act,) = agent.step([sta], [obs])
                char = np.random.choice(64, p=act)
                output.append(char)
                if char in (1, 2, 3):
                    break
                obs = one_hot[char]
            print(repr(decode(output)))

        self.trajectories = trajectories
        self.render = render

def encoding(charset):
    import numpy as np

    charset = bytes(charset, encoding="ascii", errors="ignore")
    charmap = np.zeros(256, dtype=np.int8)
    for i, c in enumerate(charset):
        charmap[c] = i

    def encode(seq):
        need_space = False
        out = []
        for b in bytes(seq, encoding="ascii", errors="ignore"):
            b = charmap[b]
            if b == 0:
                need_space = True
            else:
                if len(out) > 0 and need_space:
                    out.append(0)
                out.append(b)
                need_space = False
        return out

    def decode(seq):
        out = []
        for b in seq:
            b = int(b)
            assert b >= 0 and b < len(charset)
            out.append(charset[b])
        return str(bytes(out), encoding="ascii")

    return encode, decode
