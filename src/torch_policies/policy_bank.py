from typing import Mapping, List
import os
import torch
from torch import nn

from .learning_params import LearningParameters
from .policy_dqn import DQN
from .policy_base import Policy
from .policy_constant import ConstantPolicy
from .network import get_MLP

POLICY_MODULES: Mapping[str, Policy] = {
    "dqn": DQN
}


class PolicyBank:
    """
    This class includes a list of policies (a.k.a neural nets) for achieving different LTL goals
    """
    def __init__(self, sess, num_actions, num_features, learning_params: LearningParameters, policy_type="dqn", device="cpu"):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params

        self.device = device

        self.policies: List[Policy] = []
        self.policy2id: Mapping[str, int] = {}

        self._add_constant_policy("False", 0.0)
        self._add_constant_policy("True", 1/learning_params.gamma)  # this ensures that reaching 'True' gives reward of 1
        self.policy_type = policy_type

        self.optimizer: torch.optim.Optimizer = None


    def _add_constant_policy(self, ltl, value):
        policy = ConstantPolicy(ltl, value, self.num_actions, device=self.device)
        self._add_policy(ltl, policy)

    def _add_policy(self, ltl, policy):
        self.policy2id[ltl] = len(self.policies)
        self.policies.append(policy)
    
    def _init_optimizer(self, parameters):
        self.optimizer = torch.optim.Adam(parameters, lr=1e-4)
    
    def add_LTL_policy(self, ltl, f_task, dfa, load_tf=True):
        """
        Add new LTL policy to the bank
        """
        if ltl not in self.policy2id:
            PolicyModule = POLICY_MODULES[self.policy_type]
            # get the network
            nn_module = get_MLP(
                self.num_features, self.num_actions, 
                hidden_layers=[64], 
                device=self.device)
            
            # initialize and add the policy module
            policy = PolicyModule(ltl, f_task, dfa, nn_module, 
                                  self.num_features, self.num_actions, self.learning_params,
                                  device=self.device)
            self._add_policy(ltl, policy)

            # add parameters to the optimizer
            if self.optimizer is None:
                self._init_optimizer(policy.parameters())
            else:
                self.optimizer.add_param_group({'params': policy.parameters()})
    
    def replace_policy(self, ltl, f_task, dfa):
        print("replace")
        # TODO is it needed or maybe we should remove
        PolicyModule = POLICY_MODULES[self.policy_type]
        nn_module = get_MLP(
            self.num_features, self.num_actions,
            hidden_layers=[64, 64], 
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

    def learn(self, s1, a, s2, next_goals, r=None, terminated=None):
        """
        given the sampled batch, computes the loss and learns the policy
        next goals is a list of next goals for each item.
        """
        assert self.optimizer is not None, "Optimizer is not initialized. Please add a policy first."
        C = len(self.policies)
        N, S = s1.shape
        A = self.num_actions
        s1_NS = torch.tensor(s1, dtype=torch.float32, device=self.device)
        a_N = torch.tensor(a, dtype=torch.int64, device=self.device)
        s2_NS = torch.tensor(s2, dtype=torch.float32, device=self.device)
        if r is None:
            r_N = torch.zeros((N,), dtype=torch.float32, device=self.device)
        else:
            r_N = torch.tensor(r, dtype=torch.float32, device=self.device)
        if terminated is None:
            terminated_N = torch.zeros((N,), dtype=torch.bool, device=self.device)
        else:
            terminated_N = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        next_goal_NC = torch.tensor(next_goals, dtype=torch.int64, device=self.device)
        
        next_q_values_CNA = torch.zeros((C, N, A), device=self.device)
        for i, policy in enumerate(self.policies):
            next_q_values_CNA[i, :, :] = policy.forward(s2_NS)
        
        self.optimizer.zero_grad()
        loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for idx, policy in enumerate(self.policies[2:]): # compute loss for every policy except for true, false
            loss += policy.compute_loss(
                s1_NS, a_N, s2_NS, r_N, terminated_N, 
                next_goal_NC[:, idx], 
                next_q_values_CNA)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def get_best_action(self, ltl, s1):
        return self.policies[self.policy2id[ltl]].get_best_action(s1)

    def update_target_network(self):
        for i in range(self.get_number_LTL_policies()):
            self.policies[i+2].update_target_network()

    def get_policy_next_LTL(self, ltl, true_props):
        return self.policies[self.get_id(ltl)].dfa.progress_LTL(ltl, true_props)

        
    def load_bank(self, policy_bank_prefix):
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        checkpoint = torch.load(checkpoint_path)
        for ltl, policy_id in self.policy2id.items():
            policy: nn.Module = self.policies[policy_id]
            policy.load_state_dict(checkpoint['policies'][ltl])
        self.optimizer.load_state_dict(checkpoint['optim'])
        print("loaded policy bank from", checkpoint_path)


    def save_bank(self, policy_bank_prefix):
        # TODO
        save = {}
        policies_dict = {}
        for ltl, policy_id in self.policy2id.items():
            policy: nn.Module = self.policies[policy_id]
            state_dict = policy.state_dict()
            policies_dict[ltl] = state_dict
        save['policies'] = policies_dict
        save['optim'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(policy_bank_prefix, "policy_bank.pth")
        torch.save(save, checkpoint_path)
        print("Saved bank to", checkpoint_path)
