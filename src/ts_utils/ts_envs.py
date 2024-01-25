from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv, DummyVectorEnv
from envs.game_creator import get_game
from envs.miniworld.params import GameParams


NUM_PARALLEL_JOBS = 11

def generate_envs(game_name="miniworld_simp_no_vis", map_id=13, parallel=False, seed=0):
    if not parallel:
        test_envs = DummyVectorEnv(
            [lambda: get_game(name=game_name, params=GameParams(
                map_fpath=f"../experiments/maps/map_{map_id}.txt",
                ltl_task=("until", "True", "a"),
                # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
                prob=1
            ), 
            max_episode_steps=1500, do_transpose=False, 
            reward_scale=10, ltl_progress_is_term=True, no_info=True)]
        )
        # test_envs = ShmemVectorEnv(
        #     [lambda: get_game(name=game_name, params=GameParams(
        #         map_fpath=f"../experiments/maps/map_{map_id}.txt",
        #         ltl_task=("until", "True", "a"),
        #         # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
        #         prob=1
        #     ) ,max_episode_steps=1500, do_transpose=False, reward_scale=10, ltl_progress_is_term=True, no_info=True) \
        #         for _ in range(NUM_PARALLEL_JOBS)]
        # )
        train_envs = DummyVectorEnv(
            [lambda: get_game(name=game_name, params=GameParams(
                map_fpath=f"../experiments/maps/map_{map_id}.txt",
                ltl_task=("until", "True", "a"),
                # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
                prob=1
            ), 
            max_episode_steps=1500, do_transpose=False, 
            reward_scale=10, ltl_progress_is_term=True, no_info=True)]
        )
    else:
        test_envs = ShmemVectorEnv(
            [lambda: get_game(name=game_name, params=GameParams(
                map_fpath=f"../experiments/maps/map_{map_id}.txt",
                ltl_task=("until", "True", "a"),
                # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
                prob=1
            ) ,max_episode_steps=1500, do_transpose=False, reward_scale=10, ltl_progress_is_term=True, no_info=True) \
                for _ in range(NUM_PARALLEL_JOBS)]
        )
        train_envs = ShmemVectorEnv(
            [lambda: get_game(name=game_name, params=GameParams(
                map_fpath=f"../experiments/maps/map_{map_id}.txt",
                ltl_task=("until", "True", "a"),
                # ltl_task=("until", "True", ("and", "a", ("until", "True", "b"))),
                prob=1
            ) ,max_episode_steps=1500, do_transpose=False, reward_scale=10, ltl_progress_is_term=True, no_info=True) \
                for _ in range(NUM_PARALLEL_JOBS)]
        )
    train_envs.seed(seed)
    test_envs.seed(seed)
    return train_envs, test_envs
