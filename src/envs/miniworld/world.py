from gymnasium import spaces, utils
from typing import Mapping, Tuple
import numpy as np

from miniworld.entity import COLOR_NAMES, Ball, Box, Key, Entity
from miniworld.miniworld import MiniWorldEnv

from .params import GameParams


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
AGENT_MARKER = "A"
OBSTACLE_MARKER = "X"
BLOCK_SCALE = 0.5 # in meters

def mat_to_opengl(i, j, num_rows, offset=0.5):
    offset *= BLOCK_SCALE
    return (j + offset, num_rows - i - 1 + offset)

def get_map_size(self, map_mat):
    width = 0
    height = 0
    for i, l in enumerate(map_mat):
        # I don't consider empty lines!
        l_stripped = l.rstrip()
        if len(l.rstrip()) == 0: continue

        # this is not an empty line!
        height += 1
        width = max(width, len(l_stripped))
    return width, height

def get_map_obj_set(self, map_mat):
    item_set = set()
    for l in map_mat:
        l_stripped = l.rstrip()
        if len(l.rstrip()) == 0: continue
        for e in l.rstrip():
            if e in OBJ_MAP.keys():
                item_set.add(e)
    return item_set


class NewPickupObjects(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +1 when agent picked up object

    ## Arguments

    ```python
    PickupObjects(size=12, num_objs=5)
    ```

    `size`: size of world

    `num_objs`: number of objects

    """

    def __init__(self, params: GameParams, **kwargs):
        with open(params.map_fpath, 'r') as f:
            map_mat = f.readlines()
        width, height = get_map_size(map_mat)
        size = max(width, height)
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self, max_episode_steps=400, **kwargs)
        utils.EzPickle.__init__(self, size, map_mat, **kwargs)

        self._map_mat = map_mat

        # create a new observation space
        obj_set = get_map_obj_set(map_mat)
        num_features = len(obj_set)
        max_dist = np.sqrt(self.size ** 2)
        self.observation_space = spaces.Box(
            low=np.zeros((1 * num_features,)),
            high=np.ones((1 * num_features,)) * max_dist
        )

    def _gen_world(self):
        self._load_map(self._map_mat)
    
    def _load_map(self, map_mat):
        """
        This method adds the entities and agents to the game
        The inputs:
            - map_fpath: path to the map file
        
        Converts matrix units to OpenGL x-z coord.
        """
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        # contains all the actions that the agent can perform
        items_set = set()
        height = 0
        width = 0
        # loading the map
        for i, l in enumerate(map_mat):
            # I don't consider empty lines!
            l_stripped = l.rstrip()
            if len(l.rstrip()) == 0: continue

            # this is not an empty line!
            height += 1
            for j, e in enumerate(l_stripped):
                if e in OBJ_MAP.keys():
                    Entity, color = OBJ_MAP[e]
                    entity = Entity(color=color)

                    # set obstacle to be static (non-pickable)
                    if e == OBSTACLE_MARKER:
                        entity.is_static = True
                    
                    # place the item
                    x, z = mat_to_opengl(i, j)
                    self.place_entity(entity, min_x=x, max_x=x, min_z=z, max_z=z)

                    # update the feature set
                    items_set.add(e)
                elif e == AGENT_MARKER:
                    # add the agent
                    x, z = mat_to_opengl(i, j)
                    self.place_agent(min_x=x, max_x=x, min_z=z, max_z=z)
            width = max(width, len(l_stripped))
        return items_set

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_per_objs:
                termination = True

        return obs, reward, termination, truncation, info
