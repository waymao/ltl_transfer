from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from copy import deepcopy
from miniworld.miniworld import MiniWorldEnv

from ltl.dfa import *
from .params import GameParams
from .constants import OBJ_REV_MAP, get_ent_str, HIT_PROP_PARAMS


class ProgressionTerminateWrapper(gym.Wrapper):
    def __init__(self, env: MiniWorldEnv, params: GameParams, reward_scale=1):
        """
        Wraps around the miniworld env, adding necessary LTL-related func.
        """
        super().__init__(env)
        self.params = params
        self.prob = self.params.prob
        self.env = env
        self.last_dfa_state = None
        self.reward_scale = reward_scale

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.last_dfa_state = deepcopy(info['dfa_state'])
        if options is not None and options.get('task_params', None) is not None:
            self.params: GameParams = options['task_params']
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, game_over, trunc, info = super().step(action)

        # in addition to game overs, if we got a progression we also consider it a game over
        # if it's not game over, and not last state, and not failure state (-1)
        if not game_over and info['dfa_state'] != self.last_dfa_state and info['dfa_state'] != -1:
            game_over = True
            rew = self.params.succ_rew * self.reward_scale
        self.last_dfa_state = deepcopy(info['dfa_state'])
        return obs, rew, game_over, trunc, info

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
    
    def reset(self, *, seed=None, options=None):
        if options is not None and options.get('task_params', None) is not None:
            self.params = options['task_params']
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
            "dfa_state": self.dfa.state,
            "loc": self.unwrapped.curr_state,
            "traversed_edge": None
        }
        return obs, info
    
    def step(self, action):
        obs, rew, ter, trunc, info = self.env.step(action)
        
        # progress
        prev_state = self.dfa.state
        true_props = self.get_true_propositions()
        self.dfa.progress(true_props)
        # assign the correct reward
        rew = self.params.succ_rew if self.dfa.in_terminal_state() else \
              self.params.step_rew
        self.ltl_game_over = self.dfa.is_game_over()
        self.env_game_over = ter
        if self.do_transpose:
            obs = np.transpose(obs, (2, 0, 1))
        
        # collect LTL related info
        info = {
            **info,
            "true_props": true_props,
            "ltl_goal": self.get_LTL_goal(),
            "dfa_game_over": self.ltl_game_over,
            "dfa_state": self.dfa.state,
            "loc": self.unwrapped.curr_state,
            "self_edge": self.dfa.nodelist[prev_state][prev_state],
            "traversed_edge": self.dfa.nodelist[prev_state][self.dfa.state]
        }
        return obs, rew * self.reward_scale, self.ltl_game_over or ter, trunc, info

    def get_true_propositions(self) -> str:
        """
        Returns the string with the propositions that are True in this state
        """
        test_pos = self.unwrapped.agent.pos + \
            HIT_PROP_PARAMS['test_dist_scale'] * \
            self.unwrapped.agent.dir_vec * self.unwrapped.agent.radius
        ent = self.unwrapped.intersect(
            self.unwrapped.agent, 
            test_pos, HIT_PROP_PARAMS['test_radius_scale'] * self.unwrapped.agent.radius
        )
        symbol = get_ent_str(ent)
        return symbol.replace("X", "")

    def get_LTL_goal(self) -> tuple:
        """
        Returns the next LTL goal
        """
        return self.dfa.get_LTL()
