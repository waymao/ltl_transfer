class LearningParameters:
    def __init__(self, lr=0.0001, max_timesteps_per_task=50000, buffer_size=25000,
                print_freq=1000, exploration_fraction=0.1, exploration_final_eps=0.02,
                train_freq=1, batch_size=32, learning_starts=1000, gamma=0.9,
                target_network_update_freq=100):
        """Parameters
        -------
        lr: float
            learning rate for adam optimizer
        max_timesteps_per_task: int
            number of env steps to optimizer for per task
        buffer_size: int
            size of the replay buffer
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
            set to None to disable printing
        batch_size: int
            size of a batched sampled from replay buffer for training
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        """
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
        self.target_network_update_freq = target_network_update_freq
