
from . import BaseWorld

class Gym(BaseWorld):
    def __init__(self, env_name, *, max_steps=None):
        import gym
        import gym.spaces
        import numpy as np

        get_env, return_env = env_manager(
            lambda: env_name() if callable(env_name)
            else gym.make(str(env_name))
        )

        def space_size(s):
            if isinstance(s, gym.spaces.Box):
                return np.prod(s.shape)
            elif isinstance(s, gym.spaces.Discrete):
                return s.n
            else:
                raise ValueError("Unsupported space: %s" % s)

        # Build one copy of the environment just to check shapes
        env = get_env()
        obs_space = env.observation_space
        obs_size = space_size(obs_space)
        act_space = env.action_space
        act_size = space_size(act_space)
        return_env(env)
        del env

        def process_obs(o):
            if isinstance(obs_space, gym.spaces.Box):
                return np.reshape(o, (obs_size,))
            elif isinstance(obs_space, gym.spaces.Discrete):
                one_hot = np.zeros(obs_size)
                one_hot[int(o)] = 1.0
                return one_hot
            else:
                raise ValueError("Unsupported space: %s" % obs_space)

        def process_action(a):
            assert a.shape == (act_size,)
            if isinstance(act_space, gym.spaces.Box):
                if not hasattr(act_space, "diff"):
                    act_space.diff = act_space.high - act_space.low
                    assert (act_space.diff > 0.001).all()
                    assert (act_space.diff < 1000).all()
                a = np.reshape(a, act_space.shape)
                a = np.abs((a - 1.0) % 4.0 - 2.0) * 0.5
                return act_space.diff * a + act_space.low
            elif isinstance(act_space, gym.spaces.Discrete):
                i = np.argmax(a)
                assert a[i] >= 0.99
                assert a[i] <= 1.01
                return i
            else:
                raise ValueError("Unsupported space: %s" % act_space)

        def trajectories(agent, n):
            # Avoid creating too many copies of the environment
            if n > 64:
                result = []
                while len(result) < n:
                    result += trajectories(
                        agent,
                        min(n - len(result), 64)
                    )
                return result

            envs = [get_env() for _ in range(n)]
            trajs = [[] for _ in envs]

            obs = [process_obs(e.reset()) for e in envs]
            obs_idx = range(len(envs))

            if agent.n_states >= 1:
                obs = np.concatenate(
                    (obs, np.zeros((n, agent.n_states))),
                    axis=1
                )

            while len(obs) >= 1:
                # Interrupt this loop after max_steps
                if len(trajs[obs_idx[0]]) == max_steps:
                    break

                # Ask the agent to process observations
                act = agent.outputs(obs)
                assert len(act) == len(obs_idx)
                act = np.reshape(act, (len(act), -1))

                # Do a step in each environment and gather observations
                next_obs, next_obs_idx = [], []
                for a, o, i in zip(act, obs, obs_idx):
                    state = a[act_size:]
                    a = a[:act_size]
                    next_o, r, done, _ = envs[i].step(process_action(a))
                    next_o = process_obs(next_o)
                    trajs[i].append((o[:obs_size], a, float(r)))
                    if not done:
                        next_o = np.concatenate((next_o, state))
                        next_obs.append(next_o)
                        next_obs_idx.append(i)

                obs, obs_idx = next_obs, next_obs_idx

            for e in envs:
                return_env(e)

            return trajs

        # Always use the same environments for rendering
        get_render_env, return_render_env = env_manager(get_env)

        def render(agent):
            env = get_render_env()
            state = np.zeros(agent.n_states)
            obs = process_obs(env.reset())
            obs = np.concatenate((obs, state))
            env.render()
            n_steps = 0

            done = False
            while not done:
                (act,) = agent.outputs([obs])
                act = np.reshape(act, -1)
                state = act[act_size:]
                act = act[:act_size]

                obs, rew, done, _ = env.step(process_action(act))
                env.render()

                obs = process_obs(obs)
                obs = np.concatenate((obs, state))

                n_steps += 1
                if n_steps == max_steps:
                    break

            return_render_env(env)

        self.trajectories = trajectories
        self.render = render

def env_manager(make_env):
    import threading

    free_envs = []
    lock = threading.Lock()

    def get_env():
        with lock:
            if len(free_envs) >= 1:
                return free_envs.pop()

        return make_env()
        if callable(env_name):
            return env_name()
        else:
            return gym.make(str(env_name))

    def return_env(e):
        with lock:
            free_envs.append(e)

    return get_env, return_env
