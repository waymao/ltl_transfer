from collections import defaultdict
from gymnasium import utils, spaces
from gymnasium.spaces import Box
import numpy as np
from typing import Optional

from .wrapper import get_ent_str

from .world import NavigateEnv


class NavigateNoVisEnv(NavigateEnv):
    def __init__(self, *args, **kwargs):
        # tmp max val so there's no complaints about the observation space
        self.dist_max = self.angle_max = 10

        super().__init__(*args, **kwargs)
        num_objs = len([None for entity in self.unwrapped.entities if get_ent_str(entity) not in "AX"])

        dist_min = np.zeros(num_objs)
        self.dist_max = np.ones(num_objs) * np.sqrt(self.unwrapped.size ** 2)
        angle_min = np.ones(num_objs) * -np.pi / 2
        self.angle_max = np.ones(num_objs) * np.pi / 2
        self.observation_space = Box(
            low=np.stack([dist_min, angle_min], axis=1).reshape(-1),
            high=np.stack([self.dist_max, self.angle_max], axis=1).reshape(-1),
        )


    def observation(self, _):
        # generates angle and direction to any object in the world
        agent = self.agent
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
        dist /= self.dist_max
        # angle starts from the right size, + is counter clockwise
        angles = np.arctan2(x_list - agent_x, -(y_list - agent_y)) - agent.dir # OpenGL has y axis flipped
        angles = np.mod(angles, 2*np.pi) - np.pi
        angles /= self.angle_max

        return np.stack([dist, angles], axis=1).reshape(-1)
    
    def render_obs(self):
        return self.observation(None)



class LidarNoVisEnv(NavigateNoVisEnv):
    def __init__(self, *args, num_bins: int=8, **kwargs):
        self.entities_pos_dict = None
        self.num_bins = num_bins
        super().__init__(*args, **kwargs)

        self.dist_max = np.sqrt(2) * self.unwrapped.size
        
        self.observation_space = Box(
            low=0, high=self.dist_max, 
            shape=(len(self.entities_pos_dict.keys()) * self.num_bins,)
        )

        self._build_pos_dict()
    
    def _build_pos_dict(self):
        entities = defaultdict(lambda: [])
        for entity in self.unwrapped.entities:
            if get_ent_str(entity) not in "AX":
                x, _, y = entity.pos
                entities[get_ent_str(entity)].append((x, y))
        self.entities_pos_dict = {}
        for key, item in entities.items():
            self.entities_pos_dict[key] = np.array(item)
        self.entities_pos_dict = dict(sorted(self.entities_pos_dict.items(), key=lambda x: x[0]))
        # print(self.entities_pos_dict)
    
    def observation(self, _):
        # generates angle and direction to any object in the world
        if self.entities_pos_dict is None:
            self._build_pos_dict()
            return np.zeros(len(self.entities_pos_dict.keys()) * self.num_bins)
        obs = np.ones(len(self.entities_pos_dict.keys()) * self.num_bins) * self.dist_max
        agent_x, _, agent_y = self.unwrapped.agent.pos
        agent_dir = self.unwrapped.agent.dir
        agent_pos_np = np.array([agent_x, agent_y])
        for i, entity_pos_arr in enumerate(self.entities_pos_dict.values()):
            dist = np.sqrt(np.sum((entity_pos_arr - agent_pos_np) ** 2, axis=1))
            angle = (np.arctan2(
                -(entity_pos_arr[:, 1] - agent_y), # OpenGL has y axis flipped
                entity_pos_arr[:, 0] - agent_x)) - agent_dir
            angle = np.mod(angle, 2*np.pi)
            # print(i, entity_pos_arr[0, 1], entity_pos_arr[0, 0], dist, angle, (np.arctan2(
                # entity_pos_arr[:, 1] - agent_y,
                # entity_pos_arr[:, 0] - agent_x)))
            slot_len = (2 * np.pi / self.num_bins)
            for item in zip(dist, angle):
                bin_idx = (int(item[1] / slot_len + 0.5)) % self.num_bins
                obs[i * self.num_bins + bin_idx] = min(obs[bin_idx], np.min(item[0]))
        obs /= self.dist_max
        return obs


