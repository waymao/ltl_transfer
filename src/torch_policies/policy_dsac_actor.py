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
                 pi_module,
                 device="cpu",
        ):
        # note: do not use soft_max in pi_module
        super().__init__()
        self.device = device
        self.pi = pi_module

    def forward(self, x):
        return self.pi(x)

    def get_action(self, x, deterministic=False):
        # taken directly from cleanrl/cleanrl/sac_continuous_action.py
        # i don't think the logprob part is intuitive
        # logits_NA = F.softmax(self(x), dim=1)
        logits_NA = self(x)
        policy_dist = Categorical(logits=logits_NA)
        if deterministic:
            # use argmax for deterministic action during eval.
            action_N = logits_NA.argmax(dim=1)
        else:
            # use sampling for stochastic action during training.
            action_N = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs_NA = policy_dist.probs
        entropy = policy_dist.entropy()
        return action_N, entropy, action_probs_NA
