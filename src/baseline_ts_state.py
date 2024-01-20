
# %%
import os
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

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger

NUM_PARALLEL_JOBS = 4

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
class DummyPreProcess(nn.Module):
    def __init__(self, output_dim=4):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x, state):
        return x, state
state_dim = train_envs.observation_space.shape[0]
preprocess_net = DummyPreProcess(output_dim=state_dim)
actor = Actor(preprocess_net, 4, [256, 256, 256], device="cuda")
actor_optim = Adam(actor.parameters(), lr=1e-4)
critic1 = Critic(preprocess_net, hidden_sizes=[256, 256, 256], last_size=4, device="cuda")
critic1_optim = Adam(critic1.parameters(), lr=1e-4)
critic2 = Critic(preprocess_net, hidden_sizes=[256, 256, 256], last_size=4, device="cuda")
critic2_optim = Adam(critic2.parameters(), lr=1e-4)

policy = DiscreteSACPolicy(
    actor, actor_optim, 
    critic1, critic1_optim, 
    critic2, critic2_optim,
    alpha=0.03,
    gamma=0.99
).to("cuda")

# logger
run_name = input("run name: ")
writer = SummaryWriter(f"results/tianshou/16-{run_name}/")
logger = TensorboardLogger(writer)


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
    logger=logger
)

trainer.run()

# %%



