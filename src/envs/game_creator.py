# from .grid.game import Game as GridGame
# from .miniworld import NavigateEnv, MiniWorldLTLWrapper
# from typing import Union

def get_game(name, params):
    if name == "grid":
        from .grid.game import Game as GridGame
        return GridGame(params)
    elif name == "miniworld":
        from .miniworld import NavigateEnv, MiniWorldLTLWrapper
        env = NavigateEnv(params)
        env = MiniWorldLTLWrapper(env, params)
        return env         
    else:
        raise ValueError(f"Unknown game: {name}")
