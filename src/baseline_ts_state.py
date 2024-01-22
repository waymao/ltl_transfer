
# %%
import os

from ts_utils.ts_policy_bank import create_discrete_sac_policy
os.environ["PYGLET_HEADLESS"] = "true"
os.environ["PYGLET_HEADLESS_DEVICE"] = "0"

# %%
from envs.game_creator import get_game
from envs.miniworld.params import GameParams
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
import numpy as np

import time

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger

NUM_PARALLEL_JOBS = 4

device = "cpu"

# %%
test_envs = get_game(name="miniworld_simp_no_vis", params=GameParams(
    map_fpath="../experiments/maps/map_13.txt",
    ltl_task=("until", "True", "a"),
    # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
    prob=1
), max_episode_steps=1000, do_transpose=False, reward_scale=10)
train_envs = get_game(name="miniworld_simp_no_vis", params=GameParams(
    map_fpath="../experiments/maps/map_13.txt",
    ltl_task=("until", "True", "a"),
    # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
    prob=1
), max_episode_steps=1000, do_transpose=False, reward_scale=10)
# test_envs = SubprocVectorEnv(
#     [lambda: get_game(name="miniworld", params=GameParams(
#         map_fpath="../experiments/maps/map_16.txt",
#         ltl_task=("until", "True", "a"),
#         # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
#         prob=1
#     ), max_episode_steps=1000, do_transpose=True) for _ in range(NUM_PARALLEL_JOBS)]
# )
# train_envs = SubprocVectorEnv(
#     [lambda: get_game(name="miniworld", params=GameParams(
#         map_fpath="../experiments/maps/map_16.txt",
#         ltl_task=("until", "True", "a"),
#         # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
#         prob=1
#     ), max_episode_steps=1000, do_transpose=True) for _ in range(NUM_PARALLEL_JOBS)]
# )

# %%
# model = DQN("CnnPolicy", game, 
#             verbose=1,
#             target_update_interval=1000,
#             learning_starts=10000,
#             train_freq=1)

# logger
run_name = time.time()
writer = SummaryWriter(f"results/tianshou/16-{run_name}/")
logger = TensorboardLogger(writer)


policy = create_discrete_sac_policy(
    num_actions=test_envs.action_space.n, 
    num_features=test_envs.observation_space.shape[0], 
    hidden_layers=[256, 256, 256],
    device=device
)


train_buffer = ReplayBuffer(int(1e6))
train_collector = Collector(policy, train_envs, train_buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)
trainer = OffpolicyTrainer(
    policy, 
    train_collector, 
    test_collector, 
    max_epoch=100, 
    step_per_epoch=5000,
    episode_per_test=20, 
    batch_size=64, 
    update_per_step=1,
    step_per_collect=12,
    logger=logger,
    stop_fn=lambda x: x >= 9, # mean test reward
)

trainer.run()

# %%



