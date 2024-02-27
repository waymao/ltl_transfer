import gymnasium as gym

class ReseedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
    
    def seed(self, seed=None):
        return self.env.reset(seed=seed)

class NoInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, {}

    def step(self, action):
        obs, rew, ter, trunc, info = super().step(action)
        return obs, rew, ter, trunc, {}
