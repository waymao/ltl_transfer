from typing import Type
import numpy as np
import dataclasses
import argparse

@dataclasses.dataclass(init=True, repr=True)
class LearningParameters:
    lr: float = 1e-4
    max_timesteps_per_task: int = 100000
    buffer_size: int = int(2e5)
    print_freq: int = 5000
    train_freq: int = 1
    batch_size: int = 32
    learning_starts: int = 5000
    gamma: int = 0.9
    max_timesteps_per_episode: int = 1000

    # DQN
    exploration_fraction: int = 0.2
    exploration_final_eps: int = 0.02
    

    # SAC related
    target_network_update_freq: int = 4
    pi_lr: float = 1e-4
    alpha: float = 0.05
    tau: float = 0.005
    auto_alpha: bool = False
    target_entropy: float = -.3 # target entropy for SAC, used to compute alpha
    non_active_target_entropy: float = -.3 # target entropy for non-active SAC policies
    dsac_random_steps: int = 0 # take random actions for this amount of time

    # CNN related
    cnn_shared_net: bool = False # use shared CNNs for all policies' actor and critics
    goal_conditioned: bool = False # use goal conditioned policies for all policies' actor and critics

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     print("Using Learn parameters:", str(self))



def get_learning_parameters(policy_name, game_name, **kwargs):
    # remove none items
    kwargs = {key: val for (key, val) in kwargs.items() if val is not None}
    
    if policy_name == "dsac":
        if 'alpha' in kwargs and kwargs['alpha'] == None:
            del kwargs['alpha']
        if game_name == "miniworld" or game_name == "miniworld_no_vis":
            params = LearningParameters(
                gamma=0.99,
                alpha=0.03,
                batch_size=512,
                tau=1, # tau per update
                lr=1e-4,
                pi_lr=1e-5,
                print_freq=5000,
                learning_starts=30000,
                train_freq=12,
                target_entropy=0.2 * -np.log(1.0 / 4),
                non_active_target_entropy=0.2 * -np.log(1.0 / 4),
                target_network_update_freq=1000,
                max_timesteps_per_episode=1000,
                max_timesteps_per_task=1500000,
                cnn_shared_net=True
            )
        else:
            params = LearningParameters(**kwargs)
    elif policy_name == "dqn":
        if game_name == "miniworld":
            params = LearningParameters(
                lr=1e-4,
                max_timesteps_per_task=200000,
                train_freq=4,
                batch_size=32,
                learning_starts=50000,
                exploration_fraction=0.1,
                exploration_final_eps=0.05,
                target_network_update_freq=10000,
                gamma=0.99,
                print_freq=5000,
            )
        elif game_name == "miniworld_no_vis":
            params = LearningParameters(
                lr=1e-4,
                max_timesteps_per_task=300000,
                buffer_size=25000,
                train_freq=1,
                batch_size=32,
                learning_starts=10000,
                exploration_fraction=0.3,
                target_network_update_freq=100,
            )
        else:
            params = LearningParameters(
                lr=1e-4,
                max_timesteps_per_task=50000,
                buffer_size=25000,
                train_freq=1,
                batch_size=32,
                learning_starts=1000,
                exploration_fraction=0.15,
                target_network_update_freq=100,
            )
    for key, val in kwargs.items():
        if hasattr(params, key) and val is not None:
            setattr(params, key, val)
    return params



# argparse boolean shenanigans generated by chatgpt4, 12/18/2023
class BooleanAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ['true', 't', 'yes', 'y', '1']:
            setattr(namespace, self.dest, True)
        elif values.lower() in ['false', 'f', 'no', 'n', '0']:
            setattr(namespace, self.dest, False)
        else:
            raise ValueError("Invalid boolean value")
# end chatgpt4 generated code

def add_fields_to_parser(parser: argparse.ArgumentParser, DataClass: Type[LearningParameters], prefix=""):
    for field in dataclasses.fields(DataClass):
        if field.type == bool:
            parser.add_argument(
                f"--{prefix}{field.name}", 
                type=str,
                action=BooleanAction, 
                help=f"type: {field.type.__name__}, auto added from: {DataClass.__name__}")
            # parser.add_argument(
            #     f"--{prefix}{field.name}", 
            #     type=str, 
            #     required=False,
            #     default=None,
            #     help=f"type: {field.type.__name__}; orig default: {field.default}; auto added from: {DataClass.__name__}")
        else:
            parser.add_argument(
                f"--{prefix}{field.name}", 
                type=field.type, 
                required=False,
                default=None,
                help=f"type: {field.type.__name__}; orig default: {field.default}; auto added from: {DataClass.__name__}")

