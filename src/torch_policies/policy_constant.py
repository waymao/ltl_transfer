import torch
from torch import nn

from .policy_base import Policy 

class ConstantPolicy(nn.Module, metaclass=Policy):
    def __init__(self, 
                 ltl, 
                 dfa, 
                 state_dim, 
                 action_dim, 
                 value=1, 
                 device="cpu"):
        self.ltl, self.value = ltl, value
        self.q_target_value = torch.ones((action_dim), dtype=torch.float32, device=device) * value
        self.device = device
        super().__init__()
    
    def forward(self, x):
        N = x.shape[0]
        return torch.tile(self.q_target_value, (N, 1))
    
    def get_v(self, x):
        N = x.shape[0]
        return torch.ones((N,), dtype=torch.float32, device=self.device) * self.value

    def compute_loss(self, s1_NS, a_N, s2_NS, r_N, terminated_N, next_q_index_N, next_q_values_CNA):
        return 0
    
    def get_state_dict(self):
        # nothing to save
        return {}
    
    def restore_from_state_dict(self, state_dict):
        # nothing to restore
        pass
