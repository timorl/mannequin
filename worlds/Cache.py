
from . import BaseWorld

class Cache(BaseWorld):
    def __init__(self, max_size=None):
        import numpy as np

        all_data = []

        def add_trajectory(*trajs):
            nonlocal all_data

            for t in trajs:
                obs, act, rew = zip(*t)

                # Use vertical arrays to save some memory
                obs = np.array(obs, dtype=np.float32)
                act = np.array(act, dtype=np.float32)
                rew = np.array(rew, dtype=np.float32)
                assert rew.shape == (len(t),)

                # Prevent writing to slices
                obs.setflags(write=False)
                act.setflags(write=False)
                rew.setflags(write=False)

                all_data.append((obs, act, rew))

            if max_size is not None:
                if len(all_data) > max_size:
                    all_data = all_data[-max_size:]

        rng = np.random.RandomState()

        def trajectories(agent, n):
            assert agent == None
            assert n >= 1

            if len(all_data) < 1:
                raise ValueError("Experience cache is empty")

            return [
                [(o, a, r) for o, a, r in zip(*all_data[i])]
                for i in rng.choice(len(all_data), size=n)
            ]

        self.add_trajectory = add_trajectory
        self.trajectories = trajectories
