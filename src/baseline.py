
# %%
import os
os.environ["PYGLET_HEADLESS"] = "true"
os.environ["PYGLET_HEADLESS_DEVICE"] = "0"

# %%
from envs.game_creator import get_game
from envs.miniworld.params import GameParams

# %%
game = get_game(name="miniworld", params=GameParams(
    map_fpath="../experiments/maps/map_16.txt",
    ltl_task=("until", "True", "a"),
    # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
    prob=1
), max_episode_steps=1000, do_transpose=False)

# %%
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env


# %%
# model = DQN("CnnPolicy", game, 
#             verbose=1,
#             target_update_interval=1000,
#             learning_starts=10000,
#             train_freq=1)
model = PPO("CnnPolicy", game, 
            verbose=1)
model.learn(total_timesteps=250000)
model.save("ppo_cartpole")

# %%



