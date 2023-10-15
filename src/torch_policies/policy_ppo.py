# Implementation of PPO
# Inspired by https://hrl.boyuai.com/
# Implements algorithm shown at
#     https://spinningup.openai.com/en/latest/algorithms/ppo.html


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .policy_base import Policy

def compute_adv(gamma: float, lmbda: float, delta_N: torch.Tensor) -> torch.Tensor:
    # Computes the generalized advantage estimator according to 
    # http://arxiv.org/abs/1506.02438
    device = delta_N.device
    delta_N = delta_N.detach().numpy()
    N = delta_N.shape[0]
    curr_adv = 0
    adv_list = np.zeros_like(delta_N)
    for i, delta in enumerate(reversed(delta_N)):
        curr_adv = gamma * lmbda * curr_adv + delta
        adv_list[N - i - 1] = curr_adv
    return torch.from_numpy(adv_list).to(device)


class PPO(nn.Module):
    """
    PPO Module.
    """
    # NOTE: v == critic. pi == actor.
    def __init__(self, 
                 ltl,
                 f_task, # full task
                 dfa,
                 actor_module: nn.Module, 
                 critic_module: nn.Module, 
                 learning_parameters,
                 logger,
                 device="cpu"):
        super().__init__()

        # DFA / LTL related stuff. 
        # TODO: maybe we can modularize it out into an "option" so we can plug in other RL policies.
        self.dfa = dfa
        self.ltl = ltl
        self.f_task = f_task

        # extract params TODO allow pass in
        gamma = learning_parameters.gamma
        lmbda = 0.9
        lr_actor = 1e-5
        lr_critic = 2e-5
        eps=0.2
        update_per_train=4
        kl_earlystop=0.2

        # RL stuff
        self.actor = actor_module
        self.critic = critic_module
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # similar to TD-lambda, or TD-n. sliding window ratio.
        self.device = device
        self.pi_optim = torch.optim.Adam(params=actor_module.parameters(), lr=lr_actor)
        self.v_optim = torch.optim.Adam(params=critic_module.parameters(), lr=lr_critic)
        self.eps = eps  # clip limit
        self.kl_earlystop = kl_earlystop    # kl threshold for early stopping
        self.update_per_train = update_per_train    # number of passes per train
    
    def forward(self, x):
        x = torch.from_numpy(x).to(self.device)
        logits = self.actor(x)
        action_list = torch.distributions.Categorical(logits)
        action = action_list.sample()
        return action.item()
    
    def get_best_action(self, x):
        # alias
        return self.forward(x)
    
    def get_v(self, x):
        return self.critic(x).squeeze(1)
    
    def learn(
            self, 
            s1_NS,
            a_N,
            s2_NS,
            rew_N,
            terminated_N,
            next_v_id_N,
            next_v_vals_CN
        ):
        a_N1 = a_N.view(-1, 1)

        with torch.no_grad():
            # compute advantage
            pi_old_N = self.actor.forward(s1_NS).gather(1, a_N1)
            v_old_val_N = self.critic(s1_NS)[0]
            # use the correct V-value given the LTL.
            v_old_next_val_N = torch.gather(next_v_vals_CN, 0, next_v_id_N.view(1, -1)).squeeze(0)
            v_old_tgt_N = rew_N + self.gamma * v_old_next_val_N * ~terminated_N
            delta_N = v_old_tgt_N - v_old_val_N
            adv_N = compute_adv(self.gamma, self.lmbda, delta_N)

        for _ in range(self.update_per_train):
            ### Update of Actor
            # compute new policy output
            self.pi_optim.zero_grad()
            pi_new_N = self.actor.forward(s1_NS).gather(1, a_N1)
            
            # ratio
            ratio_N = pi_new_N / pi_old_N
            kl = torch.sum(pi_new_N * torch.log(ratio_N))
            if self.kl_earlystop is not None and kl > self.kl_earlystop:
                # avoid going too far from the last update
                # print("stopped @", _)
                break
            # print("kl:", kl.item())
            clip_adv_N = torch.clamp(ratio_N, 1 - self.eps, 1 + self.eps) * adv_N
            pi_loss = -torch.min(ratio_N * adv_N, clip_adv_N).mean()
            # print(actor_loss.item())
            pi_loss.backward()
            self.pi_optim.step()

            ### update of critic
            self.v_optim.zero_grad()
            v_loss = torch.mean(F.mse_loss(self.critic(s1_NS).squeeze(1), v_old_tgt_N))
            v_loss.backward()
            self.v_optim.step()
    
    def get_save_state_dict(self):
        """
        custom func to get the state dict including the optim
        """
        return {
            "state": self.state_dict(),
            "actor_optim": self.v_optim.state_dict(),
            "critic_optim": self.pi_optim.state_dict()
        }
    
    def restore_from_state_dict(self, state_dict):
        """
        custom func to restore the state dict including the optim
        """
        self.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.pi_optim.load_state_dict(state_dict["critic_optim"])
        

    def get_edge_labels(self):
        """
        Return proposition formula representing outgoing edges, e.g. a & b
        """
        return self.dfa.nodelist[self.dfa.ltl2state[self.ltl]].values()

    # def add_initiation_set_classifier(self, edge, classifier):
    #     self.edge2classifier[edge] = classifier
