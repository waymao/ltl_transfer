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
from .policy_bank_cnn import PolicyBankCNN
from .rl_logger import RLLogger
from .network import get_CNN_Dense, get_CNN_preprocess, get_MLP

POLICY_MODULES: Mapping[str, Policy] = {
    "dqn": DQN,
    "ppo": PPO,
    "dsac": DiscreteSAC
}

HIDDEN_LAYER_SIZE = [64, 64]
REWARD_SCALE = 10

class PolicyBankCNNShared(PolicyBankCNN):
    """
    This class includes a list of policies (a.k.a neural nets) for achieving different LTL goals
    """
    def __init__(self, 
                 num_actions, num_features, 
                 learning_params: LearningParameters, 
                 policy_type="dqn", 
                 separate_cnn_ac=True, # whether to use separate CNN for actor and critic
                 device="cpu"):
        super().__init__(num_actions, num_features, learning_params, policy_type, device)
        self.cnn_preprocess = None
        self.separate_cnn_ac = separate_cnn_ac

    def add_LTL_policy(self, ltl, f_task, dfa, load_tf=True):
        """
        Add new LTL policy to the bank
        """
        if ltl in self.policy2id:
            return
        # CNN shared preprocess net
        if self.cnn_preprocess is None:
            self.cnn_preprocess = get_CNN_preprocess(3, device=self.device)
            self.cnn_preprocess_target = deepcopy(self.cnn_preprocess)
            self.cnn_preprocess_target.eval()
            if self.separate_cnn_ac:
                self.cnn_preprocess_q2 = get_CNN_preprocess(3, device=self.device)
                self.cnn_preprocess_q2_target = deepcopy(self.cnn_preprocess_q2)
                self.cnn_preprocess_q2_target.eval()
                self.cnn_preprocess_pi = get_CNN_preprocess(3, device=self.device)
            self.preprocess_out_dim = 1536
        if self.rl_algo == "dsac":
            pi_module = get_CNN_Dense(
                self.cnn_preprocess,
                self.preprocess_out_dim,
                out_dim=self.num_actions,
                device=self.device
            )
            critic_module = get_CNN_Dense(
                self.cnn_preprocess,
                self.preprocess_out_dim,
                out_dim=self.num_actions,
                device=self.device)
            critic1_tgt = critic_module.deepcopy_w_preprocess(self.cnn_preprocess_target)

            # depending on whether we use separate CNN for actor and critic
            # we use separate CNN for critic2 and actor
            if self.separate_cnn_ac:
                critic2_module = get_CNN_Dense(
                    self.cnn_preprocess_q2,
                    self.preprocess_out_dim,
                    out_dim=self.num_actions,
                    device=self.device)
                critic2_tgt = critic2_module.deepcopy_w_preprocess(self.cnn_preprocess_q2_target)
                pi_module = get_CNN_Dense(
                    self.cnn_preprocess_pi,
                    self.preprocess_out_dim,
                    out_dim=self.num_actions,
                    device=self.device
                )
            else:
                critic2_module = get_CNN_Dense(
                    self.cnn_preprocess,
                    self.preprocess_out_dim,
                    out_dim=self.num_actions,
                    device=self.device)
                critic2_tgt = critic2_module.deepcopy_w_preprocess(self.cnn_preprocess_target)
                pi_module = get_CNN_Dense(
                    self.cnn_preprocess,
                    self.preprocess_out_dim,
                    out_dim=self.num_actions,
                    device=self.device
                )
            actor_module = DiscreteSoftActor(
                self.num_features, self.num_actions,
                pi_module,
                device=self.device
            )
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
                auto_alpha=self.learning_params.auto_alpha,
                # TODO dsac random steps
                device=self.device
            )
        else:
            nn_module = get_CNN_Dense(
                self.cnn_preprocess,
                self.preprocess_out_dim,
                out_dim=self.num_actions,
                device=self.device)
            nn_module_tgt = nn_module.deepcopy_w_preprocess(self.cnn_preprocess_target)

            # initialize and add the policy module
            policy = DQN(ltl, f_task, dfa, nn_module, 
                                self.num_features, self.num_actions, self.learning_params,
                                self.logger,
                                nn_module_tgt=nn_module_tgt,
                                device=self.device)
        self._add_policy(ltl, policy)

        
    def load_bank(self, policy_bank_prefix, verbose=False):
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        checkpoint = torch.load(checkpoint_path)
        for ltl, policy_id in self.policy2id.items():
            if ltl not in checkpoint['policies']:
                if ltl != "True" and ltl != "False":
                    print("Warning: policy", ltl, " in bank not found in checkpoint")
                continue # skip unsaved policies
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
