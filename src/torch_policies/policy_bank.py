from typing import Mapping, List
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
    def __init__(self, num_actions, num_features, learning_params: LearningParameters, policy_type="dqn", device="cpu"):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params

        self.policies: List[Policy] = []
        self.policy2id: Mapping[str, int] = {}

        self._add_constant_policy("False", 0.0)
        self._add_constant_policy("True", 1/learning_params.gamma)  # this ensures that reaching 'True' gives reward of 1
        self.policy_type = policy_type

        self.device = device
        self.optimizer: torch.optim.Optimizer = None


    def _add_constant_policy(self, ltl, value):
        policy = ConstantPolicy(ltl, value, self.num_features)
        self._add_policy(ltl, policy)

    def _add_policy(self, ltl, policy):
        self.policy2id[ltl] = len(self.policies)
        self.policies.append(policy)
    
    def _init_optimizer(self, parameters):
        self.optimizer = torch.optim.Adam(parameters, lr=1e-4)
    
    def add_LTL_policy(self, ltl, f_task, dfa):
        """
        Add new LTL policy to the bank
        """
        if ltl not in self.policy2id:
            PolicyModule = POLICY_MODULES[self.policy_type]
            # get the network
            nn_module = get_MLP(self.num_features, self.num_actions, hidden_layers=[64, 64])
            
            # initialize and add the policy module
            policy = PolicyModule(ltl, f_task, dfa, nn_module, self.num_features, self.num_actions, self.learning_params)
            self._add_policy(ltl, policy)

            # add parameters to the optimizer
            if self.optimizer is None:
                self.optimizer = self._init_optimizer(policy.parameters())
            else:
                self.optimizer.add_param_group(policy.parameters())
    
    def replace_policy(self, ltl, f_task, dfa):
        print("replace")
        # TODO is it needed or maybe we should remove
        PolicyModule = POLICY_MODULES[self.policy_type]
        nn_module = get_MLP(self.num_features, self.num_actions, hidden_layers=[64, 64])
        policy = PolicyModule(ltl, f_task, dfa, nn_module, self.num_features, self.num_actions, self.learning_params)
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

    def learn(self, s1, a, s2, r, terminated, next_goal):
        """
        given the sampled batch, computes the loss and learns the policy
        """
        assert self.optimizer is not None, "Optimizer is not initialized. Please add a policy first."
        C = self.get_number_LTL_policies()
        N, S = s1.shape
        A = self.num_actions
        s1_NS = torch.tensor(s1, dtype=torch.float32, device=self.device)
        a_N = torch.tensor(a, dtype=torch.int64, device=self.device)
        s2_NS = torch.tensor(s2, dtype=torch.float32, device=self.device)
        r_N = torch.tensor(r, dtype=torch.float32, device=self.device)
        terminated_N = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        next_goal_N = torch.tensor(next_goal, dtype=torch.int64, device=self.device)
        
        next_q_values_CNA = torch.zeros((C, N, A))
        for i, policy in enumerate(self.policies):
            next_q_values_CNA[i, :, :] = policy.forward(s2_NS)
        
        self.optimizer.zero_grad()
        loss = 0
        for policy in self.policies:
            loss += policy.compute_loss(s1_NS, a_N, s2_NS, r_N, terminated_N, next_goal_N, next_q_values_CNA)
        loss.backward()
        self.optimizer.step()
        
        pass
