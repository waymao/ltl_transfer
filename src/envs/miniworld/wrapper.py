import gymnasium as gym
import numpy as np
from miniworld.miniworld import MiniWorldEnv

from ltl.dfa import *
from .params import GameParams
from .constants import OBJ_REV_MAP, get_ent_str

class MiniWorldDistObsWrapper(gym.Wrapper):
    pass


class MiniWorldLTLWrapper(gym.Wrapper):
    def __init__(self, env: MiniWorldEnv, params: GameParams, do_transpose=False, reward_scale=1):
        """
        Wraps around the miniworld env, adding necessary LTL-related func.
        """
        super().__init__(env)
        self.params = params
        self.prob = self.params.prob
        self.env = env
        self.do_transpose = do_transpose
        self.reward_scale = reward_scale

        # DFA / LTL
        self.dfa = DFA(self.params.ltl_task, self.params.init_dfa_state)

        if self.do_transpose:
            original_shape = self.unwrapped.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(original_shape[2], original_shape[0], original_shape[1])
            )
    
    def reset(self, *, seed=None, options=None):
        self.dfa = DFA(self.params.ltl_task, self.params.init_dfa_state)
        self.env_game_over = False
        self.ltl_game_over = False
        obs, info = super().reset(seed=seed, options=options)
        if self.do_transpose:
            obs = np.transpose(obs, (2, 0, 1))
        info = {
            **info,
            "ltl_goal": self.get_LTL_goal(),
            "dfa_game_over": self.ltl_game_over,
            "dfa_state": self.dfa.state
        }
        return obs, info
    
    def step(self, action):
        obs, rew, ter, trunc, info = self.env.step(action)
        true_props = self.get_true_propositions()
        self.dfa.progress(true_props)
        rew = 1 if self.dfa.in_terminal_state() else 0
        self.ltl_game_over = self.dfa.is_game_over()
        self.env_game_over = ter
        if self.do_transpose:
            obs = np.transpose(obs, (2, 0, 1))
        
        # collect LTL related info
        info = {
            **info,
            "ltl_goal": self.get_LTL_goal(),
            "dfa_game_over": self.ltl_game_over,
            "dfa_state": self.dfa.state
        }
        return obs, rew * self.reward_scale, self.ltl_game_over or ter, trunc, info
    
    # useful info getter for the LTL task
    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        test_pos = self.unwrapped.agent.pos + self.unwrapped.agent.dir_vec * 1.3 * self.unwrapped.agent.radius
        ent = self.unwrapped.intersect(self.unwrapped.agent, test_pos, 1.3 * self.unwrapped.agent.radius)
        symbol = get_ent_str(ent)
        return symbol.replace("X", "")

    def get_LTL_goal(self):
        """
        Returns the next LTL goal
        """
        return self.dfa.get_LTL()
    
    def get_dfa_state(self):
        return self.dfa.state
    
    def get_dfa_game_over(self):
        return self.ltl_game_over
