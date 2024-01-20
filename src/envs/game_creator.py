# from .grid.game import Game as GridGame
# from .miniworld import NavigateEnv, MiniWorldLTLWrapper
# from typing import Union
from .game_base import BaseGame
from gymnasium.wrappers import GrayScaleObservation, FrameStack
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

def get_parallel_games(name, params, render_mode=None, max_episode_steps=None, do_transpose=True, reward_scale=1) -> VecEnv:
    return SubprocVecEnv([lambda: get_game(name, params, render_mode, max_episode_steps, do_transpose, reward_scale)])

def get_game(name, params, 
             render_mode=None, max_episode_steps=None, 
             do_transpose=True, reward_scale=1, 
             ltl_progress_is_term=False) -> BaseGame:
    if name == "grid":
        from .grid.game import Game as GridGame
        return GridGame(params)
    elif name == "miniworld" or name == "miniworld_no_vis" or name == "miniworld_simp_no_vis":
        from .miniworld import NavigateEnv, MiniWorldLTLWrapper, NonVisualWrapper, NavigateNoVisEnv, ProgressionTerminateWrapper
        if name == "miniworld_simp_no_vis":
            env = NavigateNoVisEnv(params, render_mode="human", view="top")
            do_transpose = False
        elif max_episode_steps is not None:
            env = NavigateEnv(params, render_mode="human", view="top", max_episode_steps=max_episode_steps)
        else:
            env = NavigateEnv(params, render_mode="human", view="top")
        # env = GrayScaleObservation(env, keep_dim=False)
        # env = FrameStack(env, num_stack=4)
        env = MiniWorldLTLWrapper(env, params, do_transpose=do_transpose, reward_scale=reward_scale)
        if name == "miniworld_no_vis":
            env = NonVisualWrapper(env)
        if ltl_progress_is_term:
            env = ProgressionTerminateWrapper(env, params)
        return env
    else:
        raise ValueError(f"Unknown game: {name}")
