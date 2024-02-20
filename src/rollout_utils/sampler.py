import math
from gymnasium.spaces import Box
from typing import Union, List
from copy import deepcopy

class BoxSpaceIterator:
    def __init__(self, space: Box, interval: Union[List[float], float]=1):
        self.space = space
        self.prog = deepcopy(space.low)
        if type(interval) != list:
            self.interval = [interval] * len(self.prog)
        else:
            self.interval = interval
        assert len(self.interval) == len(self.prog), \
            f"Interval dim {len(self.interval)} does not match space dim {len(self.prog)}"
        

    def __iter__(self):
        return self
    
    def __next__(self):
        for i in range(len(self.prog)):
            self.prog[i] += self.interval[i]
            if self.prog[i] <= self.space.high[i]:
                return self.prog
            else:
                self.prog[i] = self.space.low[i]
        raise StopIteration

    def __len__(self):
        length = 1
        for i in range(len(self.prog)):
            # include the last element
            length *= math.floor((self.space.high[i] - self.space.low[i]) / self.interval[i]) + 1
        return length

class RandomIterator:
    def __init__(self, space: Box, num_samples: int=100):
        self.space = space
        self.num_samples = num_samples
        self.count = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        self.count += 1
        if self.count > self.num_samples:
            raise StopIteration
        return None

    def __len__(self):
        return self.num_samples
