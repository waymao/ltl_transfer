from typing import List, Mapping, Union

from torch_policies.learning_params import LearningParameters, get_learning_parameters
from tianshou.policy import BasePolicy, DiscreteSACPolicy
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import PPOPolicy, DiscreteSACPolicy, TD3Policy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, ReplayBuffer
from torch_policies.network import get_CNN_preprocess
from torch.optim import Adam
import torch
from torch import nn

class TianshouPolicyBank:
    def __init__(self):
        self.policies: List[BasePolicy] = []
        self.policy2id: Mapping[Union[tuple, str], int] = {}
    
    def add_LTL_policy(self, ltl: Union[tuple, str], policy: BasePolicy):
        if ltl not in self.policy2id:
            self.policy2id[ltl] = len(self.policies)
            self.policies.append(policy)
    
    def save(self, path):
        policy_list = {
            ltl: self.policies[id].state_dict() \
                for ltl, id in self.policy2id.items()
        }
        torch.save(policy_list, path)

    def save_ckpt(self, path):
        policy_list = {
            ltl: {
                "policy": self.policies[id].state_dict(),
                "optim": self.policies[id].optim.state_dict()
            } for ltl, id in self.policy2id.items()
        }
        torch.save(policy_list, path)
    
    def get_all_policies(self):
        return {ltl: self.policies[id] for ltl, id in self.policy2id.items()}


class DummyPreProcess(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.output_dim = dim

    def forward(self, x, state):
        return x, state

def create_discrete_sac_policy(
    num_actions: int, 
    num_features: int, 
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"), 
    device="cpu"
) -> DiscreteSACPolicy:
    preprocess_net = DummyPreProcess(dim=num_features)
    actor = Actor(preprocess_net, num_actions, hidden_layers, device=device)
    actor_optim = Adam(actor.parameters(), lr=learning_params.pi_lr)
    critic1 = Critic(preprocess_net, hidden_sizes=hidden_layers, last_size=num_actions, device=device)
    critic1_optim = Adam(critic1.parameters(), lr=learning_params.lr)
    critic2 = Critic(preprocess_net, hidden_sizes=hidden_layers, last_size=num_actions, device=device)
    critic2_optim = Adam(critic2.parameters(), lr=learning_params.lr)

    policy = DiscreteSACPolicy(
        actor, actor_optim, 
        critic1, critic1_optim, 
        critic2, critic2_optim,
        alpha=learning_params.alpha,
        gamma=learning_params.gamma,
        # TODO add more
    ).to(device)
    return policy


def load_ts_policy_bank(
    policy_bank_path: str, 
    num_actions: int, 
    num_features: int, 
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"), 
    device="cpu"
) -> TianshouPolicyBank:
    policy_bank = TianshouPolicyBank()
    policy_list = torch.load(policy_bank_path)
    for ltl, policy_state_dict in policy_list.items():
        print("Loading policy for LTL: ", ltl)
        policy = create_discrete_sac_policy(
            num_actions=num_actions, 
            num_features=num_features, 
            hidden_layers=hidden_layers,
            learning_params=learning_params, 
            device=device
        )
        policy.load_state_dict(policy_state_dict)
        policy_bank.add_LTL_policy(ltl, policy)
    return policy_bank

