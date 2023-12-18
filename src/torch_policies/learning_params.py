import numpy as np
from dataclasses import dataclass

@dataclass(init=True, repr=True)
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
    dsac_random_steps: int = 0 # take random actions for this amount of time

    # CNN related
    cnn_shared_net = False # use shared CNNs for all policies' actor and critics

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     print("Using Learn parameters:", str(self))


def get_learning_parameters(policy_name, game_name, **kwargs):
    if policy_name == "dsac":
        if 'alpha' in kwargs and kwargs['alpha'] == None:
            del kwargs['alpha']
        if game_name == "miniworld":
            return LearningParameters(
                gamma=0.99,
                # alpha=0.1d,
                batch_size=512,
                tau=0.05, # TODO this is per 12 steps, so it's actually 0.004
                lr=1e-4,
                pi_lr=1e-5,
                print_freq=5000,
                learning_starts=10000,
                train_freq=12,
                target_entropy=0.89 * -np.log(1.0 / 4),
                target_network_update_freq=12,
                max_timesteps_per_episode=1000,
                max_timesteps_per_task=500000,
                **kwargs
            )
        else:
            return LearningParameters(**kwargs)
    elif policy_name == "dqn":
        if game_name == "miniworld":
            return LearningParameters(
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
                **kwargs
            )
        elif game_name == "miniworld_no_vis":
            return LearningParameters(
                lr=1e-4,
                max_timesteps_per_task=300000,
                buffer_size=25000,
                train_freq=1,
                batch_size=32,
                learning_starts=10000,
                exploration_fraction=0.3,
                target_network_update_freq=100,
                **kwargs
            )
        else:
            return LearningParameters(
                lr=1e-4,
                max_timesteps_per_task=50000,
                buffer_size=25000,
                train_freq=1,
                batch_size=32,
                learning_starts=1000,
                exploration_fraction=0.15,
                target_network_update_freq=100,
                **kwargs
            )
