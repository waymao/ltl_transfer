import torch
from torch import nn

from .policy_base import Policy 

class ConstantPolicy(Policy):
    def __init__(self, ltl, value, action_dim, device="cpu"):
        self.ltl, self.value = ltl, value
        self.q_target_value = torch.ones((action_dim), dtype=torch.float32, device=device) * value
        self.device = device
    
    def forward(self, x):
        N, S = x.shape
        return torch.tile(self.q_target_value, (N, 1))
    
    def get_v(self, x):
        N, S = x.shape
        return torch.ones((N,), dtype=torch.float32, device=self.device) * self.value

    def compute_loss(self, s1_NS, a_N, s2_NS, r_N, terminated_N, next_q_index_N, next_q_values_CNA):
        return 0
