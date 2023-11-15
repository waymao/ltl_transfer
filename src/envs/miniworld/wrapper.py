import gymnasium as gym
import numpy as np
from miniworld.miniworld import MiniWorldEnv

from ltl.dfa import *
from .params import GameParams

class MiniWorldDistObsWrapper(gym.Wrapper):
    pass


class MiniWorldLTLWrapper(gym.Wrapper):
    def __init__(self, env: MiniWorldEnv, params: GameParams):
        """
        Wraps around the miniworld env, adding necessary LTL-related func.
        """
        super().__init__(env)
        self.params = params
        self.prob = self.params.prob

        # DFA / LTL
        self.dfa = DFA(self.params.ltl_task, self.params.init_dfa_state)

        # obs space
        obj_count = self.env.count
        self.observation_space = None
    

