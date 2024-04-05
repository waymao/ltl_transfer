from typing import Tuple
import numpy as np
import gymnasium as gym
import numpy as np
from copy import deepcopy
from miniworld.miniworld import MiniWorldEnv

from envs.grid.game import GameParams
from .wrapper import MiniWorldLTLWrapper
from .constants import HIT_PROP_PARAMS, get_ent_str

from ltl.edge_check import build_edge_one_hot, sample_edge, test_edges, expr_edges_to_str


class MiniWorldEdgeCentricWrapper(gym.Wrapper):
    def __init__(self, 
                 env: MiniWorldEnv, 
                 params: GameParams, 
                 do_transpose=False, 
                 reward_scale=1,
                 prop_list="abcdef"
        ):
        """
        Wraps around the miniworld env, adding necessary LTL-related func.
        """
        super().__init__(env)
        self.params = params
        self.prob = self.params.prob
        self.env = env
        self.do_transpose = do_transpose
        self.reward_scale = reward_scale

        self.prop_list = prop_list
        self.edge = None

        self.observation_space = gym.spaces.Dict(
            {
                "obs": env.observation_space,
                # "self_edge": gym.spaces.Box(
                #     low=0, high=1, shape=(len(self.prop_list),), dtype=np.float32
                # ),
                "out_edge_goal": gym.spaces.Box(
                    low=0, high=1, shape=(len(self.prop_list),), dtype=np.float32
                ),
                # "achieved_edge": gym.spaces.Box(
                #     low=0, high=1, shape=(len(self.prop_list),), dtype=np.float32
                # ),
            }
        )
    
    def get_observation(self, obs, true_props):
        obs_new = {}
        obs_new['obs'] = deepcopy(obs)
        # obs_new["self_edge"] = self.edge_rep_cache[0]
        obs_new["out_edge_goal"] = self.edge_rep_cache[1]
        # obs_new["achieved_edge"] = build_edge_one_hot(
        #     true_props, 
        #     self.prop_list, 
        #     absence_is_false=True
        # )
        return obs_new


    def reset(self, *, seed=None, options=None):
        if options is not None and options.get('task_params', None) is not None:
            self.params = options['task_params']
        self.env_game_over = False
        self.ltl_game_over = False
        obs, info = super().reset(seed=seed, options=options)
        if self.do_transpose:
            obs = np.transpose(obs, (2, 0, 1))
        
        # change edge if needed
        should_change_edge = np.random.rand() < 0
        if should_change_edge or self.edge is None:
            self.edge = sample_edge(self.prop_list, max_props=self.params.max_edge_props)
            self.edge_str_rep = expr_edges_to_str(*self.edge)
            self.edge_rep_cache = (
                build_edge_one_hot(self.edge_str_rep[0], self.prop_list),
                build_edge_one_hot(self.edge_str_rep[1], self.prop_list)
            )
        true_props = self.get_true_propositions()
        info = {
            **info,
            "true_props": true_props,
            "edge_goal": self.edge_str_rep,
            "loc": self.unwrapped.curr_state,
        }
        return self.get_observation(obs, true_props), info
    
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
    
    def step(self, action):
        obs, rew, ter, trunc, info = self.env.step(action)
        
        # progress
        true_props = self.get_true_propositions()
        self_satisfied, out_satisfied = test_edges(
            self.edge[0], self.edge[1], true_props
        )

        # assign the correct reward
        game_over = (not self_satisfied) or out_satisfied
        if self_satisfied:
            rew = self.params.step_rew
        elif out_satisfied:
            rew = self.params.succ_rew
        else:
            rew = self.params.fail_rew
        
        self.env_game_over = ter
        if self.do_transpose:
            obs = np.transpose(obs, (2, 0, 1))
        
        # collect LTL related info
        info = {
            **info,
            "true_props": true_props,
            "edge_goal": self.edge_str_rep,
            "loc": self.unwrapped.curr_state,
        }
        return self.get_observation(obs, true_props), rew * self.reward_scale, game_over or ter, trunc, info
    