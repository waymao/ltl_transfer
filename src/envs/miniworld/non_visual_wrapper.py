from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box
from .world import NavigateEnv
from typing import Any
from .wrapper import get_ent_str
import numpy as np

class NonVisualWrapper(ObservationWrapper):
    def __init__(self, env: NavigateEnv):
        super().__init__(env)
        num_objs = len(self.unwrapped.entities)

        dist_min = np.zeros(num_objs)
        dist_max = np.ones(num_objs) * np.sqrt(self.unwrapped.size ** 2)
        angle_min = np.ones(num_objs) * -np.pi
        angle_max = np.ones(num_objs) * np.pi
        self.observation_space = Box(
            low=np.stack([dist_min, angle_min], axis=1).reshape(-1),
            high=np.stack([dist_max, angle_max], axis=1).reshape(-1),
        )

    def observation(self, _: Any) -> Any:
        # generates angle and direction to any object in the world
        agent = self.unwrapped.agent
        agent_x, _, agent_y = agent.pos

        x_list = []
        y_list = []
        for entity in self.unwrapped.entities:
            if get_ent_str(entity) not in "AX": # ignore agents
                x, _, y = entity.pos
                x_list.append(x)
                y_list.append(y)
        
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        dist = np.sqrt((x_list - agent_x) ** 2 + (y_list - agent_y) ** 2)
        # angle starts from the right size, + is counter clockwise
        angles = (np.arctan2(x_list - agent_x, -(y_list - agent_y))) - agent.dir # OpenGL has y axis flipped
        angles = np.mod(angles, 2*np.pi) - np.pi

        return np.stack([dist, angles], axis=1).reshape(-1)
