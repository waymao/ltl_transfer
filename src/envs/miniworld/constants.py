from miniworld.entity import COLOR_NAMES, Ball, Box, Entity
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
ALWAYS_RANDOM_AGENT_LOC = True

DEFAULT_GAME_PARAMS = deepcopy(DEFAULT_PARAMS)
# stochastic parameters
# DEFAULT_GAME_PARAMS.set("forward_step", 0.2, 0.18, 0.24)
# DEFAULT_GAME_PARAMS.set("cam_fov_y", 90, 85, 95)

# deterministic parameters
DEFAULT_GAME_PARAMS.set("forward_step", 0.12, 0.12, 0.12)
DEFAULT_GAME_PARAMS.set("forward_drift", 0, 0.0, 0.0)
DEFAULT_GAME_PARAMS.set("turn_step", 15, 15, 15)
