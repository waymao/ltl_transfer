from .learning_params import LearningParameters
import torch
from abc import ABC, abstractmethod

class Policy(ABC, type(torch.nn.Module)):
    @abstractmethod
    def get_v(self, x):
        # get value function for state
        pass
    
    @abstractmethod
    def update_target_network(self) -> None:
        """Synchronize the weight for the target network."""
        pass
    
    @abstractmethod
    def compute_loss(
            self, 
            s1_NS, 
            a_N, 
            s2_NS, 
            r_N, 
            terminated_N, 
            next_q_index_N, 
            max_q_values_CN
        ):
        return 0

    @abstractmethod
    def get_edge_labels(self):
        """
        Return proposition formula representing outgoing edges, e.g. a & b
        """
        return self.dfa.nodelist[self.dfa.ltl2state[self.ltl]].values()
    
    @abstractmethod
    def get_state_dict(self) -> dict:
        pass

    @abstractmethod
    def restore_from_state_dict(self, state_dict: dict):
        pass