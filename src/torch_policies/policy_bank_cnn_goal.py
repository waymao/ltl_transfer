from typing import Mapping, List, Union, Callable, Optional
import os
import numpy as np
import torch
from torch import nn
from copy import deepcopy

from .learning_params import LearningParameters
from .policy_dqn import DQN
from .policy_ppo import PPO
from .policy_dsac import DiscreteSAC
from .policy_dsac_actor import DiscreteSoftActor
from .policy_base import DummyPolicy, Policy
from .policy_constant import ConstantPolicy
from .policy_bank import PolicyBank
from .rl_logger import RLLogger
from .network_goal_cond import get_whole_GoalCNN

POLICY_MODULES: Mapping[str, Policy] = {
    "dqn": DQN,
    "ppo": PPO,
    "dsac": DiscreteSAC
}

HIDDEN_LAYER_SIZE = [64, 64]
REWARD_SCALE = 10

class PolicyBankCNN(PolicyBank):
    """
    This class includes a list of policies (a.k.a neural nets) for achieving different LTL goals
    """
    def __init__(self, num_actions, num_features, learning_params: LearningParameters, policy_type="dqn", device="cpu"):
        super().__init__(num_actions, num_features, learning_params, policy_type, device)
        self.cnn_preprocess = None
    
    def _add_true_false_policy(self, gamma):
        self._add_constant_policy("False", 0.0)
        self._add_constant_policy("True", REWARD_SCALE/gamma)  # this ensures that reaching 'True' gives reward of 1

    def add_LTL_policy(self, ltl, f_task, dfa, load_tf=True):
        """
        Add new LTL policy to the bank
        """
        if ltl in self.policy2id:
            return
        policy = DummyPolicy(
            ltl,
            f_task,
            dfa
        )
        self._add_policy(ltl, policy)
    

    def _init_policies(self):
        pi_module = get_whole_GoalCNN(3, self.num_actions, num_policies=self.get_number_LTL_policies(), device=self.device)
        actor_module = DiscreteSoftActor(
            self.num_features, self.num_actions,
            pi_module,
            device=self.device
        )
        critic_module = get_whole_GoalCNN(3, self.num_actions, num_policies=self.get_number_LTL_policies(), device=self.device)
        critic1_tgt = deepcopy(critic_module)
        critic1_tgt.eval()
        critic2_module = get_whole_GoalCNN(3, self.num_actions, num_policies=self.get_number_LTL_policies(), device=self.device)
        critic2_tgt = deepcopy(critic2_module)
        critic2_tgt.eval()
        self.policy = DiscreteSAC(
            None,
            None, # full task
            None,
            q1=critic_module,
            q2=critic2_module,
            q1_target=critic1_tgt,
            q2_target=critic2_tgt,
            pi=actor_module,
            lr_q=self.learning_params.lr, 
            lr_pi=self.learning_params.pi_lr, 
            # lr_alpha=1e-2,
            alpha=self.learning_params.alpha,
            tau=self.learning_params.tau,
            # target_entropy=-0.89 * np.log(1 / self.num_actions),
            # target_entropy=-.3,
            state_dim=self.num_features,
            action_dim=self.num_actions,
            start_steps=self.learning_params.learning_starts,
            target_entropy=self.learning_params.target_entropy,
            secondary_target_entropy=self.learning_params.non_active_target_entropy,
            auto_alpha=self.learning_params.auto_alpha,
            # TODO dsac random steps
            device=self.device
        )


    def get_best_action(self, ltl, s1, deterministic=False):
        policy_id = self.get_id(ltl)
        return self.policy.get_best_action(s1, deterministic=deterministic, goal=policy_id)

    def update_target_network(self):
        self.policy.update_target_network()

    def get_policy_next_LTL(self, ltl, true_props):
        return self.policies[self.get_id(ltl)].dfa.progress_LTL(ltl, true_props)

    def learn(self, s1, a, s2, next_goals, r=None, terminated=None, active_policy=None):
        C = len(self.policies)
        N = s1.shape[0]
        A = self.num_actions
        s1_NS = torch.tensor(s1, dtype=torch.float32, device=self.device)
        a_N = torch.tensor(a, dtype=torch.int64, device=self.device)
        s2_NS = torch.tensor(s2, dtype=torch.float32, device=self.device)

        # all r, ter, next_goal are N * (C-2)
        if r is None:
            r_NC = torch.zeros((N, C - 2), dtype=torch.float32, device=self.device)
        else:
            r_NC = torch.tensor(r, dtype=torch.float32, device=self.device)
        r_NC *= REWARD_SCALE
        # print(r_NC.mean().item(), r_NC.max().item())
        if terminated is None:
            terminated_NC = torch.zeros((N, C - 2), dtype=torch.bool, device=self.device)
        else:
            terminated_NC = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        next_goal_NC = torch.tensor(next_goals, dtype=torch.int64, device=self.device)

        q_targets_CN = torch.zeros((C, N), device=self.device, requires_grad=False)
        with torch.no_grad():
            for i, policy in enumerate(self.policies):
                q_targets_CN[i, :] = self.policy.get_v(s2_NS, goal=i)
        
        # learn every policy except for true, false
        active_policy_metrics = None
        policy_metrics = []
        for i, policy in enumerate(self.policies[2:]): 
            is_active = (i == active_policy)
            metrics = policy.learn(
                s1_NS, a_N, s2_NS, r_NC[:, i], terminated_NC[:, i], 
                next_goal_NC[:, i], 
                q_targets_CN, is_active=is_active, goal=i)
            if is_active:
                active_policy_metrics = metrics
            policy_metrics.append(metrics)
        return active_policy_metrics, policy_metrics

        
    def load_bank(self, policy_bank_prefix):
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        checkpoint = torch.load(checkpoint_path)
        for ltl, policy_id in self.policy2id.items():
            if ltl not in checkpoint['policies']: continue # skip unsaved policies
            policy: Policy = self.policies[policy_id]
            policy.restore_from_state_dict(checkpoint['policies'][ltl])
        # self.cnn_preprocess.load_state_dict(checkpoint['cnn_preprocess'])
        print("loaded policy bank from", checkpoint_path)


    def save_bank(self, policy_bank_prefix):
        # TODO
        save = {}
        policies_dict = {}
        for ltl, policy_id in self.policy2id.items():
            policy: Union[nn.Module, Policy] = self.policies[policy_id]
            if type(policy) != ConstantPolicy:
                # only save non-constant policy
                policies_dict[ltl] = policy.get_state_dict()
        save['policies'] = policies_dict
        # save['cnn_preprocess'] = self.cnn_preprocess.state_dict()
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        if not os.path.exists(policy_bank_prefix):
            os.makedirs(policy_bank_prefix)
        torch.save(save, checkpoint_path)
        print("Saved bank to", checkpoint_path)
