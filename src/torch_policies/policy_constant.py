import torch
from torch import nn

from .policy_base import Policy 

class ConstantPolicy(Policy):
    def __init__(self, ltl, value, num_features, device="cpu"):
        self.ltl, self.value = ltl, value
        self.q_target_value = torch.ones([num_features], type=torch.float32, device=device) * value
    
    def forward(self, x):
        return self.q_target_value
