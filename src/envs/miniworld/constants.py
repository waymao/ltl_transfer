from miniworld.entity import COLOR_NAMES, Ball, Box, Key, Entity
from typing import Mapping, Tuple

OBJ_MAP: Mapping[str, Tuple[Entity, str]] = {
    "a": (Box, 'red'),
    "b": (Box, 'green'),
    "c": (Box, 'blue'),
    "d": (Ball, 'red'),
    "e": (Ball, 'green'),
    "f": (Ball, 'blue'),
    "g": (Key, 'red'),
    "h": (Key, 'green'),
    "i": (Key, 'blue'),
    "X": (Box, 'grey') # obstacle
}

OBJ_REV_MAP = {}
for key, (Module, color) in OBJ_MAP.items():
    OBJ_REV_MAP[f"{Module.__name__}_{color}"] = key

AGENT_MARKER = "A"
OBSTACLE_MARKER = "X"
BLOCK_SCALE = 0.5 # in meters
