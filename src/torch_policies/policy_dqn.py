# Implementation of DQN
# Inspired by https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95/
# Also Inspired by the code in the tianshou project.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

from utils.schedules import LinearSchedule
from .learning_params import LearningParameters
from .policy_base import Policy

"""
Default params
lr=1e-3, 
gamma=0.99, 
eps=0.05, 
target_update_freq=10, 
"""
@torch.jit.script
def compute_dqn_loss(
        gamma: float,
        q_all_values_NA,
        s1_NS, 
        a_N, 
        s2_NS, 
        r_N, 
        terminated_N, 
        next_q_index_N, 
        max_q_values_CN
    ):
    """
    Compute the loss of the model.
    s1: float32, prev state.
    a: int64, action. 
    s2: float32, next state. 
    r: float32, reward.
    next_goal: int64, index of the q value to use next.
    q_values_next_all: float32, computed Q value from all models (including itself)
    """
    # q for current state
    q_values_N = torch.gather(q_all_values_NA, 1, a_N.view(-1, 1)).squeeze() # selected Q
    
    # q for next state using target model.
    with torch.no_grad():
        max_q_values_next_N = torch.gather(max_q_values_CN, 0, next_q_index_N.view(1, -1)).squeeze(0)

    # target and loss
    q_values_hat_N = r_N + gamma * max_q_values_next_N * ~terminated_N
    # dqn_loss = F.mse_loss(q_values_N, q_values_hat_N, reduction="mean")
    # dqn_loss = torch.mean(torch.square(q_values_hat_N - q_values_N))
    # less sensitive to outliers according to stable baselines
    dqn_loss = F.smooth_l1_loss(q_values_hat_N, q_values_N, reduction="mean")

    return dqn_loss

class DQN(nn.Module, metaclass=Policy):
    """
    LPOPL variant of Naive DQN
    """
    def __init__(self, 
            ltl,
            f_task, # full task
            dfa,
            nn_module,
            state_dim, 
            action_dim, 
            learning_params: LearningParameters,
            logger,
            nn_module_tgt=None,
            max_grad_norm=10,
            device="cpu"
        ):
        super().__init__()
        self.model: nn.Module = nn_module.to(device)

        # DFA / LTL related stuff. 
        # TODO: maybe we can modularize it out into an "option" so we can plug in other RL policies.
        self.dfa = dfa
        self.ltl = ltl
        self.f_task = f_task

        # deep copy the model into the target model which will
        # only be updated once in a while
        if nn_module_tgt is None:
            self.target_model = deepcopy(nn_module).to(device)
            self.target_model.eval()
        else:
            self.target_model = nn_module_tgt
        
        self.gamma = learning_params.gamma

        num_steps = learning_params.max_timesteps_per_task
        self.eps_scheduler = LinearSchedule(
            schedule_timesteps=int(learning_params.exploration_fraction * num_steps), 
            initial_p=1.0, 
            final_p=learning_params.exploration_final_eps
        )
        self.device = device
        self.update_count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t = -1
        self.max_grad_norm = max_grad_norm
        self.training = False # used to distinguish training vs testing
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_params.lr, eps=1e-7)
    
    def forward(self, x):
        return self.target_model(x)
    
    def get_v(self, x):
        curr_q_values_NA = self.model.forward(x)
        return curr_q_values_NA.max(axis=1)[0]
    
    def get_best_action(self, x, deterministic=True):
        """
        Given x, returns the q value of every action.
        """
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).to(self.device)
        if deterministic:
            return self.target_model(x).argmax().item()
        
        # add exploration
        if self.training:
            # increment the time step used by the epsilon scheduler
            self.t += 1
        if np.random.random() < self.eps_scheduler.value(self.t):
            return np.random.randint(0, self.action_dim)
        else:
            return self.target_model(x).argmax().item()
    
    def update_target_network(self) -> None:
        """Synchronize the weight for the target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def learn(
            self, 
            s1_NS, 
            a_N, 
            s2_NS, 
            r_N, 
            terminated_N, 
            next_q_index_N, 
            next_q_values_CNA,
            is_active
        ):
        """
        train using dqn loss
        """
        self.optim.zero_grad()
        q_curr_val = self.model(s1_NS)
        loss = compute_dqn_loss(self.gamma, q_curr_val, s1_NS, a_N, s2_NS, r_N, terminated_N, next_q_index_N, next_q_values_CNA)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()
        return {"q_loss": loss.item()}
    
    def get_state_dict(self):
        """
        custom func to get the state dict including the optim
        """
        return {
            "state": self.state_dict(),
            "optim": self.optim.state_dict(),
        }
    
    def restore_from_state_dict(self, state_dict):
        """
        custom func to restore the state dict including the optim
        """
        self.load_state_dict(state_dict["state"])
        self.optim.load_state_dict(state_dict["optim"])


    def get_edge_labels(self):
        """
        Return proposition formula representing outgoing edges, e.g. a & b
        """
        return self.dfa.nodelist[self.dfa.ltl2state[self.ltl]].values()

    # def add_initiation_set_classifier(self, edge, classifier):
    #     self.edge2classifier[edge] = classifier
