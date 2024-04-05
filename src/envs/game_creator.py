# from .grid.game import Game as GridGame
# from .miniworld import NavigateEnv, MiniWorldLTLWrapper
# from typing import Union
from .game_base import BaseGame
from .generic_wrappers import NoInfoWrapper, ReseedWrapper
from gymnasium.wrappers import GrayScaleObservation, FrameStack

def get_game(name, params, 
             render_mode="human", max_episode_steps=None, 
             do_transpose=True, reward_scale=1, 
             ltl_progress_is_term=False,
             edge_centric=False,
             no_info=False,
             seed_wrapper=True) -> BaseGame:
    if name == "grid":
        from .grid.game import Game as GridGame
        return GridGame(params)
    elif name in [
            "miniworld", "miniworld_lidar", "miniworld_no_vis", "miniworld_simp_no_vis",
            "miniworld_simp_lidar"
        ]:
        from .miniworld import NavigateEnv, MiniWorldLTLWrapper, NonVisualWrapper, \
            NavigateNoVisEnv, ProgressionTerminateWrapper, LidarWrapper, LidarNoVisEnv, \
            MiniWorldEdgeCentricWrapper
        
        # base env
        if name == "miniworld_simp_no_vis":
            env = NavigateNoVisEnv(params, render_mode=render_mode, view="top", max_episode_steps=max_episode_steps)
            do_transpose = False
        elif name == "miniworld_simp_lidar":
            env = LidarNoVisEnv(params, render_mode=render_mode, view="top", max_episode_steps=max_episode_steps)
            do_transpose = False
        elif max_episode_steps is not None:
            env = NavigateEnv(params, render_mode=render_mode, view="top", max_episode_steps=max_episode_steps)
        else:
            env = NavigateEnv(params, render_mode=render_mode, view="top")
        
        if name == "miniworld_no_vis":
            env = NonVisualWrapper(env)
        elif name == "miniworld_lidar":
            env = LidarWrapper(env)
        # env = GrayScaleObservation(env, keep_dim=False)
        # env = FrameStack(env, num_stack=4)
            
        # symbolic wrapper
        if not edge_centric:
            env = MiniWorldLTLWrapper(env, params, do_transpose=do_transpose, reward_scale=reward_scale)
        else:
            env = MiniWorldEdgeCentricWrapper(env, params, do_transpose=do_transpose, reward_scale=reward_scale)
        
        if ltl_progress_is_term:
            env = ProgressionTerminateWrapper(env, params, reward_scale=reward_scale)
        if no_info:
            env = NoInfoWrapper(env)
        if seed_wrapper:
            env = ReseedWrapper(env)
        return env
    else:
        raise ValueError(f"Unknown game: {name}")
