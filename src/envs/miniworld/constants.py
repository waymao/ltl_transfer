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

OBJ_REV_MAP: Mapping[Tuple[str, str], str] = {
    ("Box", "red"): "a",
    ("Box", "green"): "b",
    ("Box", "blue"): "c",
    ("Ball", "red"): "d",
    ("Ball", "green"): "e",
    ("Ball", "blue"): "f",
    ("Key", "red"): "g",
    ("Key", "green"): "h",
    ("Key", "blue"): "i",
    ("Box", "grey"): "X"
}

AGENT_MARKER = "A"
OBSTACLE_MARKER = "X"
BLOCK_SCALE = 0.5 # in meters
