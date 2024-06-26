from copy import deepcopy
from gymnasium import utils, spaces
import numpy as np
from typing import Optional

from miniworld.entity import Box, Ball, Key, Agent
from miniworld.miniworld import MiniWorldEnv

from .params import GameParams
from .constants import OBJ_MAP, AGENT_MARKER, BLOCK_SCALE, OBSTACLE_MARKER, \
    RANDOM_COLOR_LIST, RANDOM_OBJ_TYPES, DEFAULT_MAP_SIZE, IGNORE_MAP_AGENT_LOC, \
    OBJECT_SIZE, DEFAULT_GAME_PARAMS, DEFAULT_DETERM_GAME_PARAMS

# for debug info
from pyglet.gl import glGetString, GL_VENDOR, GL_RENDERER
import ctypes

print("Renderer Vendor:", ctypes.string_at(glGetString(GL_VENDOR)).decode())
print("Renderer Hardware:", ctypes.string_at(glGetString(GL_RENDERER)).decode())

def mat_to_opengl(i, j, num_rows, offset=0.5):
    return xy_to_opengl(j, i, num_rows, offset=offset)

def xy_to_opengl(x, y, num_rows, offset=0.5):
    x = float(x)
    y = float(y)
    offset *= BLOCK_SCALE
    return (x * BLOCK_SCALE + offset, y * BLOCK_SCALE + offset)

def opengl_to_xy(coord, offset=0.5):
    # for loging TODO correct the errors
    x, _, y = coord
    offset *= BLOCK_SCALE
    return ((x - offset) / BLOCK_SCALE, (y - offset) / BLOCK_SCALE)

def get_map_size(map_mat):
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

def get_map_obj_set(map_mat):
    item_set = set()
    for l in map_mat:
        l_stripped = l.rstrip()
        if len(l.rstrip()) == 0: continue
        for e in l.rstrip():
            if e in OBJ_MAP.keys():
                item_set.add(e)
    return item_set


class NavigateEnv(MiniWorldEnv, utils.EzPickle):
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
    | 4   | pickup [optional]           |
    | 5   | drop [optional]             |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +1 when agent picked up object

    ## Arguments

    ```python
    PickupObjects(size=12, num_objs=5)
    ```

    `params`: parameters [see GameParams]. if None, a map will be randomly
    generated.

    `visit_only`: whether pick / place should be enabled.

    """

    def __init__(self, params: Optional[GameParams] = None, visit_only=True, max_episode_steps=9000, **kwargs):
        self.custom_params = params
        if params is not None and params.map_fpath is not None:
            with open(params.map_fpath, 'r') as f:
                self._map_mat = f.readlines()
            width, height = get_map_size(self._map_mat)
            size = max(width, height)
            assert size >= 2
            self.size = size
        else:
            self.size = getattr(params, 'size', DEFAULT_MAP_SIZE) if params is not None else DEFAULT_MAP_SIZE
            self.num_per_objs = getattr(params, 'num_per_objs', 2) if params is not None else 2
            self._map_mat = None

        if params.prob != 1.0:
            MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, params=DEFAULT_GAME_PARAMS, **kwargs)
        else:
            MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, params=DEFAULT_DETERM_GAME_PARAMS, **kwargs)
        utils.EzPickle.__init__(self, self.size, self._map_mat, **kwargs)

        if visit_only:
            self.action_space = spaces.Discrete(self.Actions.move_back + 1)
        # self.observation_space = spaces.Box()


    def _gen_world(self):
        """
        Generate a random world if map is not set.
        Load the map if map is present.
        """
        if self._map_mat is None:
            self._random_map()
        else:
            self._load_map(self._map_mat)
    
    def _random_map(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        for obj_type in RANDOM_OBJ_TYPES:
            for color in RANDOM_COLOR_LIST:
                for count in range(self.num_per_objs):
                    if obj_type == Box:
                        ent = self.place_entity(Box(color=color, size=OBJECT_SIZE * BLOCK_SCALE))
                    elif obj_type == Ball:
                        ent = self.place_entity(Ball(color=color, size=OBJECT_SIZE * BLOCK_SCALE))
                    elif obj_type == Key:
                        ent = self.place_entity(Key(color=color))
                    else:
                        raise NotImplementedError("Unknown object type: " + str(obj_type))
                    ent.color = color

        self.place_agent()

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

        use_map_agent_loc = not IGNORE_MAP_AGENT_LOC and self.custom_params.init_loc is None
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
                    

                    # set obstacle to be static (non-pickable)
                    if e == OBSTACLE_MARKER:
                        entity = Entity(color=color, size=1.0 * BLOCK_SCALE)
                        entity.static = True
                        entity.color = color
                    else:
                        entity = Entity(color=color, size=OBJECT_SIZE * BLOCK_SCALE)
                        entity.static = True
                        entity.color = color
                    
                    # place the item
                    x, z = mat_to_opengl(i, j, num_rows=self.size)
                    self.place_entity(entity, pos=(x, 0, z), dir=0)

                    # update the feature set
                    items_set.add(e)
                elif e == AGENT_MARKER and use_map_agent_loc:
                    # add the agent if not randomizing agent loc
                    x, z = mat_to_opengl(i, j, num_rows=self.size)
                    self.place_agent(min_x=x, max_x=x, min_z=z, max_z=z)
            width = max(width, len(l_stripped))
        
        if self.custom_params.init_loc is not None or self.custom_params.init_angle is not None:
            pos = None
            angle = None
            if self.custom_params.init_loc is not None:
                x_input, y_input = self.custom_params.init_loc
                x_opengl, z_opengl = xy_to_opengl(x_input, y_input, num_rows=self.size)
                pos = [x_opengl, 0.0, z_opengl]
            if self.custom_params.init_angle is not None:
                angle = self.custom_params.init_angle / 180 * np.pi
            self.place_entity(ent=self.agent, pos=pos, dir=angle)
        elif IGNORE_MAP_AGENT_LOC:
            # add the agent in the end
            self.place_agent()
        return items_set

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        # obs = np.transpose(obs, (2, 0, 1))

        # carrying
        # if self.agent.carrying:
        #     self.entities.remove(self.agent.carrying)
        #     self.agent.carrying = None
        #     reward = 1

        #     if self.num_picked_up == self.num_per_objs:
        #         termination = True
        return obs, reward, termination, truncation, {
            "agent_init_loc": (None, None),
            "map_size": (None, None),
            **info
        }
    
    @property
    def curr_state(self):
        agent_loc = self.agent.pos # for transfer
        agent_loc = opengl_to_xy(agent_loc)
        agent_angle = self.agent.dir / np.pi * 180
        return *agent_loc, agent_angle % 360
    
    def reset(self, options: dict=None, *args, **kwargs):
        params = options.get('task_params', None) if options is not None else None
        if params is not None and type(params) == GameParams:
            self.custom_params = deepcopy(params)
        obs, info = super().reset(*args, **kwargs)

        # change agent size
        # self.agent.radius /= 2
        # obs = np.transpose(obs, (2, 0, 1))

        agent_loc = self.agent.pos # for transfer
        agent_loc = opengl_to_xy(agent_loc)
        info = {
            "agent_init_loc": (agent_loc[0], agent_loc[1]),
            "map_size": (self.max_x, self.max_z),
            **info
        }
        return obs, info