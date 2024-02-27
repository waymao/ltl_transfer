# This code is a modified version of "https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py"
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._s1_storage = None
        self._a_storage = None
        self._s2_storage = None
        self._next_goal_storage = None
        self._maxsize = size
        self._next_idx = 0
        self.size = 0

    def __len__(self):
        return len(self._storage)

    def add(self, 
        s1: np.ndarray, a: np.ndarray, 
        s2: np.ndarray, next_goal: np.ndarray, 
        rews: np.ndarray, terminated: np.ndarray
    ):
        if self._s1_storage is None:
            s_shape = s1.shape
            a_shape = np.array(a).shape
            next_goal_shape = next_goal.shape
            self._s1_storage = np.zeros((self._maxsize, *s_shape), dtype=s1.dtype)
            self._s2_storage = np.zeros((self._maxsize, *s_shape), dtype=s2.dtype)
            self._a_storage = np.zeros((self._maxsize, *a_shape))
            self._next_goal_storage = np.zeros((self._maxsize, *next_goal_shape), dtype=np.int64)
            self._rew_storage = np.zeros((self._maxsize, *next_goal_shape), dtype=np.float32)
            self._terminated_storage = np.zeros((self._maxsize, *next_goal_shape), dtype=np.float32)

        self._s1_storage[self._next_idx] = s1
        self._a_storage[self._next_idx] = np.array(a)
        self._s2_storage[self._next_idx] = s2
        self._next_goal_storage[self._next_idx] = next_goal
        self._rew_storage[self._next_idx] = rews
        self._terminated_storage[self._next_idx] = terminated
        
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self.size < self._maxsize:
            self.size += 1
    
    def to_dict(self):
        return {
            "s1": self._s1_storage,
            "s2": self._s2_storage,
            "a": self._a_storage,
            "next_goals": self._next_goal_storage,
            "rewards": self._rew_storage,
            "terminated": self._terminated_storage,
            "max_size": self._maxsize,
            "next_idx": self._next_idx,
            "size": self.size
        }
    
    def from_dict(self, dict):
        self._s1_storage = dict["s1"]
        self._s2_storage = dict["s2"]
        self._a_storage = dict["a"]
        self._next_goal_storage = dict["next_goals"]
        self._maxsize = dict["max_size"]
        self._next_idx = dict["next_idx"]
        self._rew_storage = dict["rewards"]
        self._terminated_storage = dict["terminated"]
        self.size = dict["size"]
        # sanity check
        assert type(self._s1_storage) == np.ndarray
        assert type(self._s2_storage) == np.ndarray
        assert type(self._a_storage) == np.ndarray
        assert type(self._next_goal_storage) == np.ndarray
        assert type(self._rew_storage) == np.ndarray
        assert type(self._terminated_storage) == np.ndarray
        assert type(self._maxsize) == int
        assert type(self._next_idx) == int
        assert type(self.size) == int
        assert self._s1_storage.shape == self._s2_storage.shape
        assert self._s1_storage[0] == self._s2_storage[0] == self._a_storage[0] == self._next_goal_storage[0] == self.size
        assert self._next_idx < self._maxsize
        assert self.size <= self._maxsize


    def _encode_sample(self, idxes):
        """
        returns s1, a, next_goal, rew, terminated
        """
        return \
            self._s1_storage[idxes], \
            self._a_storage[idxes], \
            self._s2_storage[idxes], \
            self._next_goal_storage[idxes], \
            self._rew_storage[idxes], \
            self._terminated_storage[idxes]

    def sample(self, batch_size, random_samples=True):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if random_samples:
            idxes = np.random.choice(np.arange(self.size), size=batch_size, replace=False)
        else:
            idxes = range(len(self._storage) - batch_size, len(self._storage))
        return self._encode_sample(idxes)
