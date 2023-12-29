# Implementation of SAC
# Mostly inspired by Clean RL.
# Also Inspired by the code in the tianshou project, boyuai, and SpinningUp.
# it's different from sac2 in that it applies some methods to make computation
# more stable, plus that when learning pi, we used weighted sum of q instead of
# just gathering selected action.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from typing import Optional

from .policy_base import Policy
from .policy_dsac_actor import DiscreteSoftActor


class DiscreteSAC(nn.Module, metaclass=Policy):
    """
    Naive DQN
    """
    def __init__(self, 
            ltl,
            f_task, # full task
            dfa,
            q1: nn.Module, 
            q1_target: Optional[nn.Module],
            q2: nn.Module, 
            q2_target: Optional[nn.Module],
            pi: DiscreteSoftActor,
            state_dim, 
            action_dim, 
            lr_q=1e-3, 
            lr_pi=1e-4, 
            lr_alpha=1e-3,
            gamma=0.99, 
            alpha=0.01, # trade off coeff
            policy_update_freq=1, # policy network update frequency
            target_update_freq=10, # target network update frequency
            tau=0.005, # soft update ratio
            start_steps=0, # initial exploration phase, per spinning up
            target_entropy=None,
            secondary_target_entropy=None,
            auto_alpha=True,
            auto_alpha_active_only=False,
            device="cpu"
        ):
        super().__init__()

        self.device = device

        # DFA / LTL related stuff. 
        # TODO: maybe we can modularize it out into an "option" so we can plug in other RL policies.
        self.dfa = dfa
        self.ltl = ltl
        self.f_task = f_task

        # q1
        self.q1 = q1.to(device)
        if q1_target is not None:
            self.q1_target = q1_target.to(device)
        else:
            self.q1_target = deepcopy(q1).to(device)
            self.q1_target.eval()
        # q2
        self.q2 = q2.to(device)
        if q2_target is not None:
            self.q2_target = q2_target.to(device)
        else:
            self.q2_target = deepcopy(q2).to(device)
            self.q2_target.eval()
        # q optim
        self.q1_optim = torch.optim.Adam(
            self.q1.parameters(), 
            lr=lr_q, 
            eps=1e-4,
            fused=True if self.device == "cuda" else None
        )
        self.q2_optim = torch.optim.Adam(
            self.q2.parameters(), 
            lr=lr_q, 
            eps=1e-4,
            fused=True if self.device == "cuda" else None
        )
        # pi
        self.pi = pi.to(device)
        self.pi_optim = torch.optim.Adam(
            self.pi.parameters(), 
            lr=lr_pi, 
            eps=1e-4,
            fused=True if self.device == "cuda" else None
        )
        self.tau = tau

        # alpha and autotuning
        self.auto_alpha = auto_alpha
        self.auto_alpha_active_only = auto_alpha_active_only
        self.secondary_target_entropy = secondary_target_entropy
        self.log_alpha = torch.tensor(np.log(alpha), device=device)
        if auto_alpha:
            self.log_alpha.requires_grad = True
            self.target_entropy = target_entropy or -0.98 * np.log(1.0 / action_dim)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        # hyperparams
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.policy_update_freq = policy_update_freq
        self.update_count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        # initial exploration phase
        self.start_steps = start_steps
        self.step = 0
    
    def forward(self, x, deterministic=False, **kwargs):
        return self.pi.forward(x, deterministic=deterministic, **kwargs)
        
    def get_best_action(self, x, deterministic=False, **kwargs):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).to(self.device)
        if self.step <= self.start_steps and not deterministic:
            self.step += 1
            return int(np.floor(np.random.rand() * self.action_dim))
        else:
            return self.forward(x, deterministic=False, **kwargs)[0][0].item()
    
    def update_target_network(self) -> None:
        """Synchronize the weight for the target network."""
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.copy_(self.tau * param + (1 - self.tau) * target_param)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    def get_v(self, s2_NS, **kwargs):
        alpha = torch.exp(self.log_alpha)
        _, entropy_N, prob_NA = self.forward(s2_NS, **kwargs)
        q1_next_NA = self.q1_target(s2_NS)
        q2_next_NA = self.q2_target(s2_NS)
        q_min_next_NA = torch.min(q1_next_NA, q2_next_NA)
        q_next_N = (prob_NA * q_min_next_NA).sum(dim=-1) + alpha * entropy_N
        return q_next_N

    def learn(
            self, 
            s1_NS,
            a_N,
            s2_NS,
            r_N,
            ter_N,
            next_id_N,
            next_v_vals_CN,
            is_active=False,
            **kwargs
        ):
        metrics = {}
        alpha = torch.exp(self.log_alpha).detach()

        # q for current state
        q1_val_N = self.q1(s1_NS, **kwargs).gather(1, a_N.view(-1, 1)).squeeze()
        q2_val_N = self.q2(s1_NS, **kwargs).gather(1, a_N.view(-1, 1)).squeeze()

        # q for next state using newly sampled actions.
        with torch.no_grad():
            return_N = torch.gather(next_v_vals_CN, 0, next_id_N.view(1, -1)).squeeze()
            y_N = r_N + self.gamma * ~ter_N * return_N
            # y_N = next_y_vals_CN[next_y_id_N]

        # q target and loss
        # back propagate q loss
        q_loss1 = F.mse_loss(q1_val_N, y_N, reduction="mean")
        q_loss2 = F.mse_loss(q2_val_N, y_N, reduction="mean")
        self.q1_optim.zero_grad(set_to_none=True)
        self.q2_optim.zero_grad(set_to_none=True)
        q_loss1.backward()
        q_loss2.backward()

        # # gradient clipping
        # norm1 = torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 100)
        # norm2 = torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 100)
        # metrics['q_grad_norm1'] = norm1.item()
        # metrics['q_grad_norm2'] = norm2.item()
        # step
        metrics['q1_loss'] = q_loss1.item()
        metrics['q2_loss'] = q_loss2.item()
        self.q1_optim.step()
        self.q2_optim.step()

        # pi loss
        _, entropy_N, action_probs_NA = self.forward(s1_NS, **kwargs)
        with torch.no_grad():
            q1_NA = self.q1(s1_NS, **kwargs)
            q2_NA = self.q2(s1_NS, **kwargs)
            min_q_NA = torch.min(q1_NA, q2_NA)
        # pi_loss = -((min_q_NA - alpha * log_pi_NA) * action_probs_NA).sum(dim=-1).mean()
        pi_loss = -(torch.sum(min_q_NA * action_probs_NA, axis=-1) + alpha * entropy_N).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        
        # # grad clipping
        # norm_pi = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1).detach()
        # metrics['pi_grad_norm'] = norm_pi.item()

        # loss
        metrics['pi_loss'] = pi_loss.item()
        self.pi_optim.step()
        metrics['pi_entropy'] = entropy_N.mean().item()

        if self.auto_alpha and (not self.auto_alpha_active_only or is_active) and self.step >= self.start_steps:
            target_entropy = self.target_entropy if is_active else self.secondary_target_entropy
            alpha_loss = torch.mean(
                    (-entropy_N + target_entropy).detach() * \
                    -torch.exp(self.log_alpha))
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha'] = torch.exp(self.log_alpha).item()
                # print(np.exp(self.log_alpha.item()))
        # print("    pi loss", pi_loss)
        # print("alpha_loss", alpha_loss)

        # update target network if necessary
        # handled by policy bank already
        # if self.update_count % self.target_update_freq == 0:
        #     self.sync_weight()
        return metrics
    
    def get_edge_labels(self):
        """
        Return proposition formula representing outgoing edges, e.g. a & b
        """
        return self.dfa.nodelist[self.dfa.ltl2state[self.ltl]].values()

    def get_state_dict(self):
        return {
            "state": self.state_dict(),
            "optim": {
                "q1": self.q1_optim.state_dict(),
                "q2": self.q2_optim.state_dict(),
                "pi": self.pi_optim.state_dict()
            }
        }
    
    def restore_from_state_dict(self, state_dict):
        self.load_state_dict(state_dict['state'])
        self.q1_optim.load_state_dict(state_dict['optim']['q1'])
        self.q2_optim.load_state_dict(state_dict['optim']['q2'])
        self.pi_optim.load_state_dict(state_dict['optim']['pi'])

    # def add_initiation_set_classifier(self, edge, classifier):
    #     self.edge2classifier[edge] = classifier
