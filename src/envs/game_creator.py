# from .grid.game import Game as GridGame
# from .miniworld import NavigateEnv, MiniWorldLTLWrapper
# from typing import Union

def get_game(name, params, render_mode=None):
    if name == "grid":
        from .grid.game import Game as GridGame
        return GridGame(params)
    elif name == "miniworld":
        from .miniworld import NavigateEnv, MiniWorldLTLWrapper
        env = NavigateEnv(params, render_mode="human", view="top")
        env = MiniWorldLTLWrapper(env, params)
        return env         
    else:
        raise ValueError(f"Unknown game: {name}")
