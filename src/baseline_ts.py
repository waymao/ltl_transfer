
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
test_envs = get_game(name="miniworld", params=GameParams(
    map_fpath="../experiments/maps/map_16.txt",
    ltl_task=("until", "True", "a"),
    # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
    prob=1
), max_episode_steps=1000, do_transpose=False, reward_scale=10)
train_envs = get_game(name="miniworld", params=GameParams(
    map_fpath="../experiments/maps/map_16.txt",
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
class TSNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, x, state):
        x = torch.Tensor(x).permute(0, 3, 1, 2).to("cuda")
        return self.net(x), state
def get_CNN_preprocess_ts(num_channels, output_dim, device="cuda"):
    preprocess_net = get_CNN_preprocess(num_channels, output_dim, device)
    return TSNet(
        preprocess_net,
    ).to("cuda")
# %%
# model = DQN("CnnPolicy", game, 
#             verbose=1,
#             target_update_interval=1000,
#             learning_starts=10000,
#             train_freq=1)
preprocess_net_actor = get_CNN_preprocess_ts(3, 1536, "cuda")
preprocess_net_critic1 = get_CNN_preprocess_ts(3, 1536, "cuda")
preprocess_net_critic2 = get_CNN_preprocess_ts(3, 1536, "cuda")
actor = Actor(preprocess_net_actor, 4, [256, 64], preprocess_net_output_dim=1536, device="cuda")
actor_optim = Adam(actor.parameters(), lr=1e-5)
critic1 = Critic(preprocess_net_critic1, hidden_sizes=[256, 64], last_size=4, preprocess_net_output_dim=1536, device="cuda")
critic1_optim = Adam(critic1.parameters(), lr=1e-4)
critic2 = Critic(preprocess_net_critic2, hidden_sizes=[256, 64], last_size=4, preprocess_net_output_dim=1536, device="cuda")
critic2_optim = Adam(critic2.parameters(), lr=1e-4)

policy = DiscreteSACPolicy(
    actor, actor_optim, 
    critic1, critic1_optim, 
    critic2, critic2_optim,
    alpha=0.05
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



