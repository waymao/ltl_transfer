from miniworld.entity import COLOR_NAMES, Ball, Box, Entity, Agent
from miniworld.params import DomainParams, DEFAULT_PARAMS
from copy import deepcopy
from .entity import Key
from typing import Mapping, Tuple

OBJ_MAP: Mapping[str, Tuple[Entity, str]] = {
    "a": (Box, 'yellow'),
    "b": (Box, 'green'),
    "c": (Box, 'blue'),
    "d": (Ball, 'yellow'),
    "e": (Ball, 'green'),
    "f": (Ball, 'blue'),
    "g": (Box, 'purple'),
    "h": (Ball, 'purple'),
    "i": (Ball, 'red'),
    "s": (Ball, 'grey'), # "shelter", but an ordinary obj in this new domain
    "X": (Box, 'grey') # obstacle
}

RANDOM_OBJ_TYPES = [Box]
RANDOM_COLOR_LIST = ["yellow", "green"]

OBJ_REV_MAP = {}
for key, (Module, color) in OBJ_MAP.items():
    OBJ_REV_MAP[f"{Module.__name__}_{color}"] = key

AGENT_MARKER = "A"
OBSTACLE_MARKER = "X"
BLOCK_SCALE = 1 # in meters TODO not working
OBJECT_SIZE = 0.7
DEFAULT_MAP_SIZE = 10
IGNORE_MAP_AGENT_LOC = True

DEFAULT_GAME_PARAMS = deepcopy(DEFAULT_PARAMS)
DEFAULT_DETERM_GAME_PARAMS = deepcopy(DEFAULT_PARAMS)
# stochastic parameters
DEFAULT_GAME_PARAMS.set("forward_step", 0.15, 0.12, 0.17)
DEFAULT_GAME_PARAMS.set("forward_drift", 0, -0.05, 0.05)
DEFAULT_GAME_PARAMS.set("turn_step", 15, 10, 20)

# deterministic parameters
DEFAULT_DETERM_GAME_PARAMS.set("forward_step", 0.15, 0.15, 0.15)
DEFAULT_DETERM_GAME_PARAMS.set("forward_drift", 0, 0.0, 0.0)
DEFAULT_DETERM_GAME_PARAMS.set("turn_step", 15, 15, 15)
# DEFAULT_DETERM_GAME_PARAMS.set("forward_step", 0.12, 0.12, 0.12)
# DEFAULT_DETERM_GAME_PARAMS.set("forward_drift", 0, 0.0, 0.0)
# DEFAULT_DETERM_GAME_PARAMS.set("turn_step", 15, 15, 15)


def get_ent_str(ent):
    if not isinstance(ent, Entity):
        return ""
    elif isinstance(ent, Agent):
        return "A"
    else:
        return OBJ_REV_MAP.get(f"{ent.__class__.__name__}_{ent.color}", "")
