import gymnasium as gym
import numpy as np
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import COLOR_NAMES, Ball, Box, Key, Entity

from ltl.dfa import *
from .params import GameParams
from .constants import OBJ_REV_MAP

class MiniWorldDistObsWrapper(gym.Wrapper):
    pass


def get_ent_str(ent):
    if not isinstance(ent, Entity):
        return ""
    else:
        return OBJ_REV_MAP.get(f"{ent.__class__.__name__}_{ent.color}", "")

class MiniWorldLTLWrapper(gym.Wrapper):
    def __init__(self, env: MiniWorldEnv, params: GameParams, is_visit=True):
        """
        Wraps around the miniworld env, adding necessary LTL-related func.
        """
        super().__init__(env)
        self.params = params
        self.prob = self.params.prob
        self.env = env

        # DFA / LTL
        self.dfa = DFA(self.params.ltl_task, self.params.init_dfa_state)

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        test_pos = self.unwrapped.agent.pos + self.unwrapped.agent.dir_vec * 1.3 * self.agent.radius
        ent = self.unwrapped.intersect(self.unwrapped.agent, test_pos, 1.1 * self.agent.radius)

        # adding the is_night proposition
        return get_ent_str(ent)

    def get_LTL_goal(self):
        """
        Returns the next LTL goal
        """
        return self.dfa.get_LTL()
