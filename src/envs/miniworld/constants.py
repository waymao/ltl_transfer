from miniworld.entity import COLOR_NAMES, Ball, Box, Entity
from .entity import Key
from typing import Mapping, Tuple

OBJ_MAP: Mapping[str, Tuple[Entity, str]] = {
    "a": (Box, 'yellow'),
    "b": (Box, 'green'),
    "c": (Box, 'blue'),
    "d": (Ball, 'yellow'),
    "e": (Ball, 'green'),
    "f": (Ball, 'blue'),
    "g": (Key, 'yellow'),
    "h": (Key, 'green'),
    "i": (Key, 'blue'),
    "X": (Box, 'grey') # obstacle
}

RANDOM_OBJ_TYPES = [Ball, Box]
RANDOM_COLOR_LIST = ["yellow", "green", "blue"]

OBJ_REV_MAP = {}
for key, (Module, color) in OBJ_MAP.items():
    OBJ_REV_MAP[f"{Module.__name__}_{color}"] = key

AGENT_MARKER = "A"
OBSTACLE_MARKER = "X"
BLOCK_SCALE = 1 # in meters
DEFAULT_MAP_SIZE = 16
