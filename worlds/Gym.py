
from . import World

class Gym(World):
    def __init__(self, env_name, max_steps=-1):
        import gym
        import gym.spaces
        import numpy as np
        import threading

        free_envs = [gym.make(str(env_name))]
        lock = threading.Lock()

        def get_env():
            with lock:
                if len(free_envs) >= 1:
                    return free_envs.pop()

            return gym.make(str(env_name))

        def return_env(e):
            with lock:
                free_envs.append(e)

        obs_space = free_envs[0].observation_space
        act_space = free_envs[0].action_space

        def process_obs(o):
            if isinstance(obs_space, gym.spaces.Box):
                return o
            elif isinstance(obs_space, gym.spaces.Discrete):
                one_hot = np.zeros(obs_space.n,)
                one_hot[int(o)] = 1.0
                return one_hot
            else:
                raise ValueError("Unsupported observation space")

        def process_action(a):
            if isinstance(act_space, gym.spaces.Box):
                if not hasattr(act_space, "diff"):
                    act_space.diff = act_space.high - act_space.low
                    assert (act_space.diff > 0.001).all()
                    assert (act_space.diff < 1000).all()
                assert a.shape == act_space.shape
                a = np.abs((a - 1.0) % 4.0 - 2.0) * 0.5
                return act_space.diff * a + act_space.low
            elif isinstance(act_space, gym.spaces.Discrete):
                assert a.shape == (act_space.n,)
                i = np.argmax(a)
                assert a[i] >= 0.99
                assert a[i] <= 1.01
                return i
            else:
                raise ValueError("Unsupported action space")

        def trajectories(agent, n):
            envs = [get_env() for _ in range(n)]
            trajs = [[] for _ in envs]

            sta = [None for _ in envs]
            obs = [process_obs(e.reset()) for e in envs]
            obs_idx = range(len(envs))

            while len(obs) >= 1:
                # Interrupt if any episode lasts longer than max_steps
                if len(trajs[obs_idx[0]]) == max_steps:
                    break

                # Ask the outer agent to process observations
                sta, act = agent.step(sta, obs)
                assert len(sta) == len(obs_idx)
                assert len(act) == len(obs_idx)

                # Do a step in each environment and gather observations
                next_sta, next_obs, next_obs_idx = [], [], []
                for s, o, a, i in zip(sta, obs, act, obs_idx):
                    next_o, r, done, _ = envs[i].step(process_action(a))
                    trajs[i].append((o, a, [r]))
                    if not done:
                        next_sta.append(s)
                        next_obs.append(process_obs(next_o))
                        next_obs_idx.append(i)

                sta, obs, obs_idx = next_sta, next_obs, next_obs_idx

            for e in envs:
                return_env(e)

            return trajs

        def render(agent):
            env = get_env()
            env.render()

            sta = [None]
            obs = process_obs(env.reset())
            done = False
            n_steps = 0
            while not done:
                sta, act = agent.step(sta, [obs])
                obs, rew, done, _ = env.step(process_action(act[0]))
                env.render()

                n_steps += 1
                if n_steps == max_steps:
                    break

            return_env(env)

        self.trajectories = trajectories
        self.render = render
