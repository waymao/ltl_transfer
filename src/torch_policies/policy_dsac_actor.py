from torch import nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

from .network import get_MLP

# with inspirations from cleanrl
# the original code uses a two-clause BSD license.
class DiscreteSoftActor(nn.Module):
    def __init__(self,
                 state_dim: int, 
                 action_dim: int, 
                 hidden=[64],
                 device="cpu",
        ):
        super().__init__()
        self.device = device
        self.pi = get_MLP(
            state_dim, 
            action_dim, 
            hidden, 
            use_relu=True, 
            device=device,
            init_method="fanin",
            output_init_scale=3e-1
        )

    def forward(self, x):
        return self.pi(x)

    def get_action(self, x):
        # taken directly from cleanrl/cleanrl/sac_continuous_action.py
        # i don't think the logprob part is intuitive
        logits_NA = self(x)
        policy_dist = Categorical(logits=logits_NA)
        action_N = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs_NA = policy_dist.probs
        log_prob = F.log_softmax(logits_NA, dim=1)
        return action_N, log_prob, action_probs_NA
