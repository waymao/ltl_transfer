import sys
from typing import List, Mapping, Optional, Tuple, Union
import json
import os
from ltl.dfa import DFA

from utils.learning_params import LearningParameters, get_learning_parameters
from tianshou.policy import BasePolicy, DiscreteSACPolicy
from tianshou.policy import PPOPolicy
from ltl.ltl_utils import convert_ltl

from classifiers import Classifier, RadiusMatcher, KNNMatcher, NNClassifier
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic
from ts_utils.network import get_whole_CNN
from torch.optim import Adam
import torch
from torch import nn

def list_to_tuple(obj):
    tmp = []
    for item in obj:
        if isinstance(item, list):
            tmp.append(list_to_tuple(item))
        else:
            tmp.append(item)
    return tuple(tmp)

class TianshouPolicyBank:
    def __init__(self):
        self.policies: List[BasePolicy] = []
        self.policy_ltls: List[str] = []
        self.classifiers: List[Optional[Classifier]] = []
        self.policy2id: Mapping[Union[tuple, str], int] = {}
        self.dfas: List[DFA] = []
        self.algo = "dsac"
    
    def add_LTL_policy(self, 
                ltl: Union[tuple, str], 
                policy: DiscreteSACPolicy, 
                classifier: Optional[Classifier] = None
        ):
        if ltl not in self.policy2id and ltl != "True" and ltl != "False":
            self.policy2id[ltl] = len(self.policies)
            self.policies.append(policy)
            self.dfas.append(DFA(ltl))
            self.policy_ltls.append(ltl)
            self.classifiers.append(classifier)
    
    def save_pb_index(self, path):
        # saving the ltl map
        with open(os.path.join(path, "ltl_list.json"), "w") as f:
            json.dump(list(self.policy2id.items()), f)
    
    def save(self, path):
        self.save_ckpt(path)
        # for ltl, id in self.policy2id.items():
        #     torch.save(self.policies[id].state_dict(), os.path.join(path, str(id) + ".pth"))

    def save_ckpt(self, path):
        os.makedirs(os.path.join(path, "policies"), exist_ok=True)
        for ltl, id in self.policy2id.items():
            if self.algo == "ppo":
                save_ppo_policy(path, id, ltl, self.policies[id])
            else:
                save_individual_policy(path, id, ltl, self.policies[id])
    
    def save_individual_policy(self, path, policy_id):
        if self.algo == "ppo":
            save_ppo_policy(path, policy_id, self.policy_ltls[policy_id], self.policies[policy_id])
        else:
            save_individual_policy(path, policy_id, self.policy_ltls[policy_id], self.policies[policy_id])
    
    def get_all_policies(self):
        return {ltl: self.policies[id] for ltl, id in self.policy2id.items()}

    def get_dfa(self, ltl):
        return self.policy2id[ltl]


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

def create_ppo_policy(
    num_actions: int, 
    num_features: int, 
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"), 
    device="cpu"
) -> DiscreteSACPolicy:
    preprocess_net = DummyPreProcess(dim=num_features)
    actor = Actor(preprocess_net, num_actions, hidden_layers, device=device)
    critic1 = Critic(preprocess_net, hidden_sizes=hidden_layers, last_size=num_actions, device=device)
    ac = ActorCritic(actor, critic1)
    optim = Adam(ac.parameters(), lr=learning_params.lr)

    policy = PPOPolicy(
        actor, 
        critic1, 
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        # discount_factor=learning_params.gamma,
        # TODO add more
    ).to(device)
    return policy


def create_vis_discrete_sac_policy(
    num_actions: int, 
    input_size: Tuple[int, int, int],
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"), 
    device="cpu"
) -> DiscreteSACPolicy:
    net_actor = get_whole_CNN(input_size[-1], num_actions, device=device)
    net_critic1 = get_whole_CNN(input_size[-1], num_actions, device=device)
    net_critic2 = get_whole_CNN(input_size[-1], num_actions, device=device)
    actor = Actor(net_actor, num_actions, device=device, softmax_output=False)
    actor_optim = Adam(actor.parameters(), lr=learning_params.pi_lr)
    critic1 = Critic(net_critic1, device=device)
    critic1_optim = Adam(critic1.parameters(), lr=learning_params.lr)
    critic2 = Critic(net_critic2, hidden_sizes=hidden_layers, last_size=num_actions, device=device)
    critic2_optim = Adam(critic2.parameters(), lr=learning_params.lr)

    policy = DiscreteSACPolicy(
        actor, actor_optim, 
        critic1, critic1_optim, 
        critic2, critic2_optim,
        alpha=learning_params.alpha,
        gamma=learning_params.gamma,
    ).to(device)
    return policy


def load_ts_policy_bank(
    policy_bank_path: str, 
    num_actions: int, 
    num_features: int, 
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"),
    load_classifier="knn_random",
    classifier_seed=0,
    eval_mode=False,
    device="cpu",
    verbose=False
) -> TianshouPolicyBank:
    policy_bank = TianshouPolicyBank()
    
    # loading the ltl-to-policy index
    with open(os.path.join(policy_bank_path, "ltl_list.json"), "r") as f:
        ltl_list = json.load(f)

    # loading each individual policy file
    for ltl, id in sorted(ltl_list, key=lambda x: x[1]):
        ltl = list_to_tuple(ltl)
        if verbose: print("Loading policy for LTL: ", convert_ltl(ltl))
        policy, ltl_stored = load_individual_policy(
            policy_bank_path, 
            id, num_actions, 
            num_features, hidden_layers, 
            learning_params, eval_mode, device)
        assert ltl == ltl_stored
        
        # loading the classifier
        try:
            if load_classifier is not None:
                classifier_name = load_classifier.split('_')
                if classifier_name[0] == "knn" or classifier_name[0] == "radius":
                    # direct data-match classifier
                    if classifier_name[0] == "random":
                        classifier = KNNMatcher()
                    else:
                        classifier = RadiusMatcher()
                    if classifier_name[1] == "uniform":
                        classifier.load(policy_bank_path, id, classifier_seed, "uniform")
                    else:
                        classifier.load(policy_bank_path, id, classifier_seed)
                elif load_classifier == "nn":
                    # neural network prediction
                    classifier = NNClassifier()
                    classifier.load(policy_bank_path, id)
                else:
                    print("Unknown classifier type", load_classifier, file=sys.stderr)
                    classifier = None
            else:
                classifier = None
        except Exception as e:
            print("Unable to load classifier for policy ", id, file=sys.stderr)
            print("   Error:", e, file=sys.stderr)
            classifier = None
        
        policy_bank.add_LTL_policy(ltl, policy, classifier)
    return policy_bank



def load_ppo_policy(
    policy_bank_path: str,
    policy_id: int,
    num_actions: int,
    num_features: int,
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"),
    eval_mode=False,
    device="cpu"
) -> Tuple[BasePolicy, str]:
    policy = create_discrete_sac_policy(
            num_actions=num_actions, 
            num_features=num_features, 
            hidden_layers=hidden_layers,
            learning_params=learning_params, 
            device=device
        )
    save_data = torch.load(os.path.join(policy_bank_path, "policies", str(policy_id) + "_ckpt.pth"))
    policy.load_state_dict(save_data['policy'])
    policy.actor_optim.load_state_dict(save_data['actor_optim'])
    policy.critic1_optim.load_state_dict(save_data['critic1_optim'])
    policy.critic2_optim.load_state_dict(save_data['critic2_optim'])
    if eval_mode:
        print(eval_mode)
        policy.eval()
    return policy, save_data['ltl']


def load_individual_policy(
    policy_bank_path: str,
    policy_id: int,
    num_actions: int,
    num_features: int,
    hidden_layers: List[int] = [256, 256, 256],
    learning_params: LearningParameters = get_learning_parameters("dsac", "miniworld_no_vis"),
    eval_mode=False,
    device="cpu"
) -> Tuple[BasePolicy, str]:
    policy = create_discrete_sac_policy(
            num_actions=num_actions, 
            num_features=num_features, 
            hidden_layers=hidden_layers,
            learning_params=learning_params, 
            device=device
        )
    save_data = torch.load(os.path.join(policy_bank_path, "policies", str(policy_id) + "_ckpt.pth"))
    policy.load_state_dict(save_data['policy'])
    policy.actor_optim.load_state_dict(save_data['actor_optim'])
    policy.critic1_optim.load_state_dict(save_data['critic1_optim'])
    policy.critic2_optim.load_state_dict(save_data['critic2_optim'])
    if eval_mode:
        print(eval_mode)
        policy.eval()
    return policy, save_data['ltl']


def save_ppo_policy(
    policy_bank_path: str,
    policy_id: int,
    ltl: str,
    policy: PPOPolicy,
):
    save_data = {
        "ltl": ltl,
        "policy": policy.state_dict(),
        "actor_optim": policy.optim.state_dict(),
    }
    torch.save(save_data, os.path.join(policy_bank_path, "policies", str(policy_id) + "_ckpt.pth"))


def save_individual_policy(
    policy_bank_path: str,
    policy_id: int,
    ltl: str,
    policy: DiscreteSACPolicy,
):
    save_data = {
        "ltl": ltl,
        "policy": policy.state_dict(),
        "actor_optim": policy.actor_optim.state_dict(),
        "critic1_optim": policy.critic1_optim.state_dict(),
        "critic2_optim": policy.critic2_optim.state_dict(),
    }
    torch.save(save_data, os.path.join(policy_bank_path, "policies", str(policy_id) + "_ckpt.pth"))


########################## TESTING ##########################
def test_load_knn_pb():
    path = "/home/wyc/data/shared/ltl-transfer-ts/results/miniworld_simp_no_vis_minecraft/mixed_p1.0/lpopl_dsac/map13/0/alpha=0.03/"
    matcher = RadiusMatcher()
    matcher.load(path, 0)
    return load_ts_policy_bank(path, 4, 22)

if __file__ == "__main__":
    test_load_knn_pb()
    print("Done!")

