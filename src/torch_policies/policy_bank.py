from typing import Mapping, List, Union, Callable, Optional
import os
import numpy as np
import torch
from torch import nn

from .learning_params import LearningParameters
from .policy_dqn import DQN
from .policy_ppo import PPO
from .policy_dsac import DiscreteSAC
from .policy_dsac_actor import DiscreteSoftActor
from .policy_base import Policy
from .policy_constant import ConstantPolicy
from .rl_logger import RLLogger
from .network import get_MLP

POLICY_MODULES: Mapping[str, Policy] = {
    "dqn": DQN,
    "ppo": PPO,
    "dsac": DiscreteSAC
}

HIDDEN_LAYER_SIZE = [256, 256, 256]
REWARD_SCALE = 10

class PolicyBank:
    """
    This class includes a list of policies (a.k.a neural nets) for achieving different LTL goals
    """
    def __init__(self, 
                 num_actions, 
                 num_features, 
                 learning_params: LearningParameters, 
                 policy_type="dqn", 
                 device="cpu"
    ):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params

        self.device = device

        self.policies: List[Policy] = []
        self.policy2id: Mapping[str, int] = {}

        self._add_true_false_policy(learning_params.gamma)
        self.rl_algo = policy_type

        self.logger = RLLogger() # TODO maybe use it to log metrics

    def _add_true_false_policy(self, gamma):
        self._add_constant_policy("False", 0.0)
        self._add_constant_policy("True", REWARD_SCALE / gamma)  # this ensures that reaching 'True' gives reward of 1

    def _add_constant_policy(self, ltl, value):
        policy = ConstantPolicy(
            ltl, 
            None,
            self.num_features,
            self.num_actions,
            value=value, 
            device=self.device
        )
        self._add_policy(ltl, policy)

    def _add_policy(self, ltl, policy):
        self.policy2id[ltl] = len(self.policies)
        self.policies.append(policy)

    def add_LTL_policy(self, ltl, f_task, dfa, load_tf=True):
        """
        Add new LTL policy to the bank
        """
        if ltl not in self.policy2id:
            PolicyModule = POLICY_MODULES[self.rl_algo]
            if self.rl_algo == "dsac":
                pi_module = get_MLP(
                    self.num_features, self.num_actions,
                    [256, 256, 256], 
                    use_relu=True, 
                    device=self.device,
                    init_method="fanin",
                    output_init_scale=3e-1
                )
                actor_module = DiscreteSoftActor(
                    self.num_features, self.num_actions,
                    pi_module,
                    device=self.device
                )
                critic_module = get_MLP(
                    num_features=self.num_features,
                    num_actions=self.num_actions,
                    hidden_layers=[256, 256, 256],
                    init_method=None
                )
                critic2_module = get_MLP(
                    num_features=self.num_features,
                    num_actions=self.num_actions,
                    hidden_layers=[256, 256, 256],
                    init_method=None
                )
                policy = DiscreteSAC(
                    ltl,
                    f_task, # full task
                    dfa,
                    q1=critic_module,
                    q1_target=None,
                    q2=critic2_module,
                    q2_target=None,
                    pi=actor_module,
                    auto_alpha=False,
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
                    device=self.device
                )
            else:    
                nn_module = get_MLP(
                    self.num_features, self.num_actions, 
                    hidden_layers=HIDDEN_LAYER_SIZE, 
                    device=self.device)
                    
                # initialize and add the policy module
                policy = PolicyModule(ltl, f_task, dfa, nn_module, 
                                    self.num_features, self.num_actions, self.learning_params,
                                    self.logger,
                                    device=self.device)
            self._add_policy(ltl, policy)
    
    def replace_policy(self, ltl, f_task, dfa):
        print("replace")
        # TODO is it needed or maybe we should remove
        PolicyModule = POLICY_MODULES[self.rl_algo]
        nn_module = get_MLP(
            self.num_features, self.num_actions,
            hidden_layers=HIDDEN_LAYER_SIZE, 
            device=self.device)
        policy = PolicyModule(
            ltl, f_task, dfa, nn_module, 
            self.num_features, self.num_actions, self.learning_params,
            device=self.device)
        self._add_policy(ltl, policy)
    
    def get_id(self, ltl):
        return self.policy2id[ltl]

    def get_LTL_policies(self):
        return set(self.policy2id.keys()) - set(['True', 'False'])

    def get_number_LTL_policies(self):
        return len(self.policies) - 2  # The '-2' is because of the two constant policies ('False' and 'True')

    def reconnect(self):
        # this function is not used for reconnecting the compute graph in the pytorch version since it's dynamic
        pass

    def learn(self, s1, a, s2, next_goals, r=None, terminated=None, active_policy=None):
        """
        given the sampled batch, computes the loss and learns the policy
        next goals is a list of next goals for each item.
        """
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
        
        # C * N
        # compute target
        q_targets_CN = torch.zeros((C, N), device=self.device, requires_grad=False)
        with torch.no_grad():
            for i, policy in enumerate(self.policies):
                q_targets_CN[i, :] = policy.get_v(s2_NS)
        
        # learn every policy except for true, false
        active_policy_metrics = None
        policy_metrics = []
        for i, policy in enumerate(self.policies[2:]): 
            is_active = (i == active_policy)
            metrics = policy.learn(
                s1_NS, a_N, s2_NS, r_NC[:, i], terminated_NC[:, i], 
                next_goal_NC[:, i], 
                q_targets_CN, is_active=is_active)
            if is_active:
                active_policy_metrics = metrics
            policy_metrics.append(metrics)
        return active_policy_metrics, policy_metrics

    def get_best_actions(self, ltl, s1, deterministic=False):
        return self.policies[self.policy2id[ltl]].get_best_actions(s1, deterministic=deterministic)
    
    def get_best_action(self, ltl, s1, deterministic=False):
        return self.policies[self.policy2id[ltl]].get_best_action(s1, deterministic=deterministic)

    def update_target_network(self):
        for i in range(self.get_number_LTL_policies()):
            self.policies[i+2].update_target_network()

    def get_policy_next_LTL(self, ltl, true_props):
        return self.policies[self.get_id(ltl)].dfa.progress_LTL(ltl, true_props)

        
    def load_bank(self, policy_bank_prefix):
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        checkpoint = torch.load(checkpoint_path)
        for ltl, policy_id in self.policy2id.items():
            if ltl not in checkpoint['policies']: continue # skip unsaved policies
            policy: Policy = self.policies[policy_id]
            policy.restore_from_state_dict(checkpoint['policies'][ltl])
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
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        if not os.path.exists(policy_bank_prefix):
            os.makedirs(policy_bank_prefix)
        torch.save(save, checkpoint_path)
        print("Saved bank to", checkpoint_path)
