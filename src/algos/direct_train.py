import os
import random
import time
from typing import Optional, Union
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
# import tensorflow as tf

from torch_policies.policy_bank import PolicyBank, LearningParameters
from torch_policies.policy_bank_cnn import PolicyBankCNN
from torch_policies.policy_bank_cnn_shared import PolicyBankCNNShared
from torch_policies.policy_bank_cnn_goal import PolicyBankCNNGoalCond

from utils.schedules import LinearSchedule
from utils.replay_buffer import ReplayBuffer
from ltl.dfa import *
from envs.game_creator import get_game
from envs.game_base import BaseGame
from test_utils import Loader, TestingParameters, load_pkl

from utils.curriculum import CurriculumLearner
from test_utils import TaskLoader, Saver
from succ_logger import SuccLogger, SuccEntry
from torch.profiler import profile, record_function, ProfilerActivity
import random
from .lpopl import _initialize_policy_bank

from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv


PROGRESSION_REW = 1
FINAL_REW = 1
STEP_REW = 0
FAIL_REW = 0

def run_experiments(
        game_name: str,
        tester: TaskLoader, 
        curriculum: CurriculumLearner, 
        saver: Saver, run_id: int, 
        num_times, incremental_steps, show_print, 
        rl_algo="dqn", resume=False, device="cpu",
        succ_log_path=None):
    time_init = time.time()
    tester_original = tester
    curriculum_original = curriculum
    curriculum.restart()
    loader = Loader(saver)
    train_dpath = os.path.join(saver.exp_dir, "train_data")

    # seeding
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)
    game = get_game(game_name, tester.get_task_params(curriculum.get_current_task()))
    # testing_games = gym.vector.AsyncVectorEnv(
    #     lambda: [get_game(game_name, tester.get_task_params(curriculum.get_current_task())) for _ in range(8)]
    # )
    testing_games = get_game(game_name, tester.get_task_params(curriculum.get_current_task()))
    game.reset(seed=run_id)

    # Running the tasks 'num_times'
    run_dpath = os.path.join(train_dpath, "run_%d" % run_id)
    # Overwrite 'tester' and 'curriculum' if incremental training
    tester_fpath = os.path.join(run_dpath, "tester.pkl")
    if resume and os.path.exists(run_dpath) and os.path.exists(tester_fpath):
        tester = load_pkl(tester_fpath)
    else:
        tester = tester_original

    # write hyperparams to test result file
    learning_params: LearningParameters = tester.learning_params
    tester.logger.add_text("learning_params", str(learning_params))
    tester.logger.add_text("run_id", str(run_id))

    # Initializing experience replay buffer
    replay_buffer = ReplayBuffer(learning_params.buffer_size)

    # Initializing policies per each subtask
    policy_bank = _initialize_policy_bank(game_name, learning_params, curriculum, tester, rl_algo=rl_algo, device=device)

    envs = get_game(game_name, tester.get_task_params(curriculum.get_current_task()))
    test_envs = get_game(game_name, tester.get_task_params(curriculum.get_current_task()))


def _run_train(
        game: SubprocVecEnv, 
        testing_game: SubprocVecEnv,
        policy_bank: PolicyBank, 
        tester: TaskLoader, curriculum: CurriculumLearner, 
        replay_buffer: ReplayBuffer, 
        show_print: bool, 
        succ_logger: SuccLogger,
        best_pb_save_dir: str,
        do_render=False,
    ):
    MAX_EPS = 1000
    # Initializing parameters
    learning_params: LearningParameters = tester.learning_params
    testing_params = tester.testing_params

    # Initializing the game
    action_space = game.action_space

    # Initializing parameters
    num_features = game.observation_space.shape[0]
    num_steps = learning_params.max_timesteps_per_task
    exploration = LinearSchedule(schedule_timesteps=int(learning_params.exploration_fraction * num_steps), initial_p=1.0, final_p=learning_params.exploration_final_eps)
    training_reward = 0

    # Starting interaction with the environment
    curr_eps_step = 0
    if show_print: print("Executing", num_steps, "actions...")
    s1, info = game.reset()
    s2 = None
    
    for t in range(0, num_steps, learning_params.parallel_envs):
        # Getting the current state and ltl goal
        ltl_goal = info['ltl_goal']

        # Choosing an action to perform
        if policy_bank.rl_algo == "dqn":
            if random.random() < exploration.value(t): a = [action_space.sample() for _ in learning_params.parallel_envs]
            else: a = policy_bank.get_best_actions(ltl_goal, s1)
        else:
            a = policy_bank.get_best_actions(ltl_goal, s1)
        
        s2, reward, term, trunc, info = game.step(a)
        if do_render:
            # print("action:", a, "true propositions:", task.get_true_propositions())
            game.render()
        training_reward += reward
        true_props = ... #TODO


    pass