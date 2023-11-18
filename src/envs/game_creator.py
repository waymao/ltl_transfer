from .grid.game import Game as GridGame
from .miniworld import NavigateEnv, MiniWorldLTLWrapper
from typing import Union

def get_game(name, params) -> Union[GridGame, MiniWorldLTLWrapper]:
    if name == "grid":
        return GridGame(params)
    elif name == "miniworld":
        env = NavigateEnv(params)
        env = MiniWorldLTLWrapper(env, params)
        return env         
    else:
        raise ValueError(f"Unknown game: {name}")
