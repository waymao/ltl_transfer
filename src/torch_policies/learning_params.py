import numpy as np

class LearningParameters:
    def __init__(self, lr=1e-4, max_timesteps_per_task=100000, buffer_size=int(2e5),
                print_freq=5000, exploration_fraction=0.2, exploration_final_eps=0.02,
                train_freq=1, batch_size=32, learning_starts=5000, gamma=0.9,
                max_timesteps_per_episode=1000,
                # SAC related
                target_network_update_freq=4,
                pi_lr=1e-4,
                alpha=0.05,
                tau=0.005,
                auto_alpha=False,
                target_entropy=-.3, # target entropy for SAC, used to compute alpha
                dsac_random_steps=0 # take random actions for this amount of time
        ):
        self.lr = lr
        self.max_timesteps_per_task = max_timesteps_per_task
        self.buffer_size = buffer_size
        self.print_freq = print_freq
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.max_timesteps_per_episode = max_timesteps_per_episode

        # SAC related
        self.target_network_update_freq = target_network_update_freq
        self.pi_lr = pi_lr
        self.alpha = alpha
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy
        self.dsac_random_steps = dsac_random_steps

        print("Using Learn parameters:", str(self))

def get_learning_parameters(policy_name, game_name, **kwargs):
    if policy_name == "dsac":
        if 'alpha' in kwargs and kwargs['alpha'] == None:
            del kwargs['alpha']
        if game_name == "miniworld":
            return LearningParameters(
                gamma=0.99,
                alpha=0.2,
                batch_size=256,
                tau=0.005,
                lr=1e-4,
                pi_lr=1e-5,
                print_freq=5000,
                learning_starts=10000,
                auto_alpha=True,
                target_entropy=0.98 * -np.log(1.0 / 4),
                target_network_update_freq=1,
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
