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
from .policy_base import Policy
from .policy_constant import ConstantPolicy
from .policy_bank import PolicyBank
from .rl_logger import RLLogger
from .network import get_CNN_Dense, get_CNN_preprocess, get_MLP, get_whole_CNN

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
        # CNN shared preprocess net
        if self.rl_algo == "dsac":
            # pi
            pi_module = get_whole_CNN(3, self.num_actions, device=self.device)
            actor_module = DiscreteSoftActor(
                self.num_features, self.num_actions,
                pi_module,
                device=self.device
            )
            critic_module = get_whole_CNN(3, self.num_actions, device=self.device, fc_layers="auto")
            critic1_tgt = deepcopy(critic_module)
            critic1_tgt.eval()
            critic2_module = get_whole_CNN(3, self.num_actions, device=self.device, fc_layers="auto")
            critic2_tgt = deepcopy(critic2_module)
            critic2_tgt.eval()
            policy = DiscreteSAC(
                ltl,
                f_task, # full task
                dfa,
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
        else:
            nn_module = get_whole_CNN(3, self.num_actions, device=self.device, fc_layers=None)
            nn_module_tgt = deepcopy(nn_module)
            nn_module_tgt.eval()

            # initialize and add the policy module
            policy = DQN(ltl, f_task, dfa, nn_module, 
                                self.num_features, self.num_actions, self.learning_params,
                                self.logger,
                                nn_module_tgt=nn_module_tgt,
                                device=self.device)
        self._add_policy(ltl, policy)

    def learn(self, s1, a, s2, next_goals, r=None, terminated=None, active_policy=None):
        return super().learn(s1, a, s2, next_goals, r, terminated, active_policy)
        
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
