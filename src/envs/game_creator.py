# from .grid.game import Game as GridGame
# from .miniworld import NavigateEnv, MiniWorldLTLWrapper
# from typing import Union
from .game_base import BaseGame
from gymnasium.wrappers import GrayScaleObservation, FrameStack

def get_game(name, params, render_mode=None, max_episode_steps=None, do_transpose=True) -> BaseGame:
    if name == "grid":
        from .grid.game import Game as GridGame
        return GridGame(params)
    elif name == "miniworld":
        from .miniworld import NavigateEnv, MiniWorldLTLWrapper
        if max_episode_steps is not None:
            env = NavigateEnv(params, render_mode="human", view="top", max_episode_steps=max_episode_steps)
        else:
            env = NavigateEnv(params, render_mode="human", view="top")
        # env = GrayScaleObservation(env, keep_dim=False)
        # env = FrameStack(env, num_stack=4)
        env = MiniWorldLTLWrapper(env, params, do_transpose=do_transpose)
        return env         
    else:
        raise ValueError(f"Unknown game: {name}")
