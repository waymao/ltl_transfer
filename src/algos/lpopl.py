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
from test_utils import Tester, Saver
from succ_logger import SuccLogger, SuccEntry
from torch.profiler import profile, record_function, ProfilerActivity
import random


PROGRESSION_REW = 1
FINAL_REW = 1
STEP_REW = 0
FAIL_REW = 0

def run_experiments(
        game_name: str,
        tester: Tester, 
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

    curriculum_fpath = os.path.join(run_dpath, "curriculum.pkl")
    if resume and os.path.exists(run_dpath) and os.path.exists(curriculum_fpath):
        curriculum = load_pkl(curriculum_fpath)
        learning_params.learning_starts += curriculum.total_steps  # recollect 'replay_buffer'
        curriculum.incremental_learning(incremental_steps)
    else:
        curriculum = curriculum_original

    # Reseting default values
    if not curriculum.incremental:
        curriculum.restart()

    # Initializing experience replay buffer
    replay_buffer = ReplayBuffer(learning_params.buffer_size)

    # Initializing policies per each subtask
    policy_bank = _initialize_policy_bank(game_name, learning_params, curriculum, tester, rl_algo=rl_algo, device=device)
    # Load 'policy_bank' if incremental training
    policy_dpath = os.path.join(saver.policy_dpath, "run_%d" % run_id)
    if resume and os.path.exists(policy_dpath) and os.listdir(policy_dpath):
        try:
            loader.load_policy_bank(policy_bank, run_id)
        except (RuntimeError, FileNotFoundError) as e:
            print()
            print("Encountered the following error when loading policy bank:")
            print(e)
            print("Will not load the bank.")
            print()
    if show_print:
        print("Total # of policies:", policy_bank.get_number_LTL_policies())
        print("Policy bank initialization took: %0.2f mins" % ((time.time() - time_init)/60))

    # Running the tasks
    num_tasks = 0
    succ_logger = SuccLogger(log_path=succ_log_path)
    try:
        while not curriculum.stop_learning():
            task = curriculum.get_next_task()
            if show_print:
                print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
                print("%d Current task: %d, %s" % (num_tasks, curriculum.current_task, str(task)))
            
            # Apply task params
            task_params = tester.get_task_params(task)
            game.reset(options=dict(task_params=task_params))
            testing_games.reset(options=dict(task_params=task_params))

            # log task
            tester.logger.add_text("task", str(task), global_step=curriculum.get_current_step())

            # Running the task
            _run_LPOPL(game, testing_games, policy_bank, tester, curriculum, replay_buffer, show_print, succ_logger, succ_log_path)
            num_tasks += 1
            # # Save 'policy_bank' for incremental training and transfer
            saver.save_policy_bank(policy_bank, run_id)
            # Backing up the results
            saver.save_results()
            # Save 'tester' and 'curriculum' for incremental training
            saver.save_train_data(curriculum, run_id)
    except KeyboardInterrupt:
        # gracefully print everything when interrupted
        # # Save 'policy_bank' for incremental training and transfer
        saver.save_policy_bank(policy_bank, run_id)
        # Backing up the results
        saver.save_results()
        # Save 'tester' and 'curriculum' for incremental training
        saver.save_train_data(curriculum, run_id)
        pass

    # Showing results
    print("Time:", "%0.2f" % ((time.time() - time_init)/60), "mins")


def _initialize_policy_bank(game_name, learning_params: LearningParameters, curriculum: CurriculumLearner, tester: Tester, load_tf=True, rl_algo="dqn", device="cpu"):
    task_aux = get_game(game_name, tester.get_task_params(curriculum.get_current_task()))
    num_actions = task_aux.action_space.n
    num_features = task_aux.observation_space.shape[0]
    if game_name != "miniworld":
        policy_bank = PolicyBank(num_actions, num_features, learning_params, policy_type=rl_algo, device=device)
    else:
        # tasks requiring visual observation
        if learning_params.goal_conditioned:
            policy_bank = PolicyBankCNNGoalCond(num_actions, num_features, learning_params, policy_type=rl_algo, device=device)
        elif learning_params.cnn_shared_net:
            policy_bank = PolicyBankCNNShared(num_actions, num_features, learning_params, policy_type=rl_algo, device=device)
        else:
            policy_bank = PolicyBankCNN(num_actions, num_features, learning_params, policy_type=rl_algo, device=device)
    for idx, f_task in enumerate(tester.get_LTL_tasks()[:tester.train_size]):  # only load first 'train_size' policies
        # start_time = time.time()
        dfa = DFA(f_task)
        # print("%d processing LTL: %s" % (idx, str(f_task)))
        # print("took %0.2f mins to construct DFA" % ((time.time() - start_time)/60))
        # start_time = time.time()
        for ltl in dfa.ltl2state:
            # this method already checks that the policy is not in the bank and it is not 'True' or 'False'
            policy_bank.add_LTL_policy(ltl, f_task, dfa, load_tf=load_tf)
        # print("took %0.2f mins to add policy" % ((time.time() - start_time)/60))
    
    # we need to initialize the entire policy if the policy is goal-conditioned
    if learning_params.goal_conditioned:
        policy_bank._init_policies()
        
    if load_tf:
        policy_bank.reconnect()  # -> creating the connections between the neural nets
    # print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank


def _run_LPOPL(
        game: BaseGame, 
        testing_game: Optional[BaseGame],
        policy_bank: PolicyBank, 
        tester: Tester, curriculum: CurriculumLearner, 
        replay_buffer: ReplayBuffer, 
        show_print: bool, 
        succ_logger: SuccLogger,
        best_pb_save_dir: str,
        do_render=False,
    ):
    if testing_game is None:
        testing_game = game # if no specific testing game, use the same game for training and testing
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

    # logging transfer info
    epi_info = SuccEntry()
    # log the starting point from which the agent is successful
    try:
        epi_info.init_x = info['agent_init_loc'][0]
        epi_info.init_y = info['agent_init_loc'][1]
        epi_info.ltl_task = info["ltl_goal"]
    except:
        epi_info = None

    print("Image shape:", s1.shape)
    print("Action shape:", game.action_space)
    s2 = None

    # aux render code for testing
    # task.render()
    # while True:
    #     input("hi")
    if do_render:
        game.render()

    epi_begin_t = 0
    active_policy_metrics = None # for logging loss
    best_succ_rate = 0
    best_mean_len = 9999999999

    for t in range(num_steps):
        # Getting the current state and ltl goal
        ltl_goal = info['ltl_goal']

        # Choosing an action to perform
        if policy_bank.rl_algo == "dqn":
            if random.random() < exploration.value(t): a = action_space.sample()
            else: a = policy_bank.get_best_action(ltl_goal, np.expand_dims(s1, axis=0))
        else:
            a = policy_bank.get_best_action(ltl_goal, np.expand_dims(s1, axis=0))
        # print(s1)
        # print(a)
        # a = int(input("action: "))
        
        # updating the curriculum
        curriculum.add_step()

        # Executing the action
        s2, reward, term, trunc, info = game.step(a)
        if do_render:
            # print("action:", a, "true propositions:", task.get_true_propositions())
            game.render()
        training_reward += reward
        true_props = game.get_true_propositions()

        # Computing rewards for each policy and saving the reward
        next_goals = np.zeros((policy_bank.get_number_LTL_policies(),), dtype=np.float64)
        rewards = np.zeros((policy_bank.get_number_LTL_policies(),), dtype=np.float64)
        terminateds = np.zeros((policy_bank.get_number_LTL_policies(),), dtype=np.float64)
        for ltl in policy_bank.get_LTL_policies():
            ltl_id = policy_bank.get_id(ltl)
            if term and reward <= 0:
                # env deadend
                next_goals[ltl_id-2] = policy_bank.get_id("False")  # env deadends are equal to achive the 'False' formula
                rewards[ltl_id-2] = FAIL_REW
                terminateds[ltl_id-2] = 1
            else:
                ltl_next_id = policy_bank.get_id(policy_bank.get_policy_next_LTL(ltl, true_props))
                next_goals[ltl_id-2] = ltl_next_id
                if ltl_next_id == ltl_id:
                    # still in the same task
                    rewards[ltl_id-2] = STEP_REW
                    terminateds[ltl_id-2] = 0
                elif ltl_next_id == policy_bank.get_id("True"):
                    # finished the task
                    rewards[ltl_id-2] = FINAL_REW
                    terminateds[ltl_id-2] = 0
                elif ltl_next_id == policy_bank.get_id("False"):
                    # ltl deadend
                    rewards[ltl_id-2] = FAIL_REW
                    terminateds[ltl_id-2] = 1
                else:
                    # progressed to a sub task
                    rewards[ltl_id-2] = PROGRESSION_REW
                    terminateds[ltl_id-2] = 0
        replay_buffer.add(s1, a, s2, next_goals, rewards, terminateds)

        # update s
        s1 = s2

        # Learning
        step = curriculum.get_current_step()

        # 20000 / 4
        if (step + 1) >= learning_params.learning_starts and (step + 1) % learning_params.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            S1, A, S2, Goal, r, terminated = replay_buffer.sample(learning_params.batch_size)

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, ) as prof:
            # with profile(activities=[ProfilerActivity.CPU]) as prof:
            #     policy_bank.learn(S1, A, S2, Goal, active_policy=curriculum.current_task)
            # # prof.export_chrome_trace("profile_trace.json")
            # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
            active_policy_metrics, policy_metrics = policy_bank.learn(
                S1, A, S2, Goal, r, terminated, active_policy=curriculum.current_task)


        # Updating the target network
        if curriculum.get_current_step() + 1 > learning_params.learning_starts and \
            (curriculum.get_current_step() + 1) % learning_params.target_network_update_freq == 0:
            # Update target network periodically.
            policy_bank.update_target_network()

        # Printing and testing
        global_step = curriculum.get_current_step() + 1
        if global_step % learning_params.print_freq == 0:
            # logging training data
            if show_print:
                print("Training Step:", global_step, "\t cum_rew:", training_reward, "\tsucc_rate:", "%0.3f"%curriculum.get_succ_rate())
            tester.logger.add_scalar(
                "train/rew",
                training_reward,
                global_step=global_step
            )
            tester.logger.add_scalar(
                "train/succ_rate",
                curriculum.get_succ_rate(),
                global_step=global_step
            )
            if active_policy_metrics is not None:
                for key, val in active_policy_metrics.items():
                    tester.logger.add_scalar(
                        f"train/active/{key}",
                        val,
                        global_step=global_step
                    )
                if show_print:
                    print("    last training policy metrics:")
                    print("       ", "; ".join([f"{key}: {val}" for key, val in active_policy_metrics.items()]))
                    print("    last all policy metrics:")
                    for i, metric in enumerate(policy_metrics):
                        print("       ", i, ":", "; ".join([f"{key}: {val}" for key, val in metric.items()]))

            # logging testing data
            # if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
            #     tester.run_test(curriculum.get_current_step(), game_name, _test_LPOPL, policy_bank, num_features)
            if global_step >= learning_params.learning_starts:
                with torch.no_grad():
                    result_dict = _test_LPOPL(testing_game, learning_params, testing_params, policy_bank)
                for key, val in result_dict.items():
                    tester.logger.add_scalar(
                        f"test/{key}",
                        val,
                        global_step=global_step
                    )
                if show_print:
                    print("Testing @ Step: {}\t ".format(global_step),
                        " ; ".join([f"{key}: {val}" for key, val in result_dict.items()]),
                    )

                # save best policy bank
                succ_rate = result_dict["succ_rate"]
                len_mean = result_dict["len_mean"]
                if succ_rate > best_succ_rate or (succ_rate == best_succ_rate and len_mean < best_mean_len):
                    policy_bank.save_bank(best_pb_save_dir)
                    print("    saved best policy bank")
                best_succ_rate = max(best_succ_rate, succ_rate)
                best_mean_len = min(best_mean_len, len_mean)
                

        # reset truncate counter if LTL was progressed. Otherwise, increment the counter
        new_ltl_goal = info["ltl_goal"]
        if new_ltl_goal != ltl_goal:
            # print("    ", curriculum.get_current_step(), ":     progressed to", new_ltl_goal, ". len:", curr_eps_step)
            curr_eps_step = 0
        else:
            curr_eps_step += 1

        # Restarting the environment (Game Over)
        if info['dfa_game_over'] or trunc or term or curr_eps_step > learning_params.max_timesteps_per_episode:
            # print("    ", curriculum.get_current_step(), 
            #       ": train game over. Final LTL:", info['ltl_goal'], 
            #       "; deadend:", (info['dfa_state'] == -1))
            curr_eps_step = 0

            if epi_info is not None:
                epi_info.success = (new_ltl_goal == "True")
                epi_info.final_ltl = new_ltl_goal
                epi_info.epi_len = t - epi_begin_t
                epi_info.global_step = curriculum.get_current_step()
                epi_info.time = time.time()
                epi_info.ltl_deadend = (info['dfa_state'] == -1)
                succ_logger.report_result(epi_info)
                succ_logger.save()

            # NOTE: Game over occurs for one of three reasons:
            # 1) DFA reached a terminal state,
            # 2) DFA reached a deadend, or
            # 3) The agent reached an environment deadend (e.g. a PIT)
            # 4) NEW: > episode max time step
            s1, info = game.reset()  # Restarting

            # reset data tracking
            if epi_info is not None:
                epi_begin_t = t + 1
                epi_info = SuccEntry()
                epi_info.init_x = info['agent_init_loc'][0]
                epi_info.init_y = info['agent_init_loc'][1]
                epi_info.ltl_task = info["ltl_goal"]

            # updating the hit rates
            curriculum.update_succ_rate(t, reward)
            if curriculum.stop_task(t) or best_succ_rate > 0.9:
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

    succ_logger.save()
    if show_print:
        tester.logger.add_scalar(
            "train/rew",
            training_reward,
            global_step=curriculum.get_current_step() + 1
        )
        tester.logger.add_scalar(
            "train/succ_rate",
            curriculum.get_succ_rate(),
            global_step=curriculum.get_current_step() + 1
        )
        print("Done! Total reward:", training_reward)


def _test_LPOPL(
        task: gym.Env, 
        learning_params: LearningParameters, 
        testing_params: TestingParameters, 
        policy_bank: PolicyBank, 
        deterministic=True,
        do_render=False
    ):
    """
    Runs the policy 'policy' on the task 'task' for 'testing_params.test_epis' episodes.
    Returns the mean and std of the accumulated rewards episode lengths, success rate, and 
    the LTL goal progression count.
    """
    task.reset(seed=testing_params.test_seed) # deterministic
    r_hist = []
    len_hist = []
    prog_count_hist = []
    succ_count = 0
    for epi in tqdm(range(testing_params.test_epis), desc="Testing...", leave=False, ascii=True):
        r_total = 0
        t = 0
        s1, info = task.reset()
        prev_ltl_goal = None
        prog_count = -1 # LTL goal progression count
        for t in range(learning_params.max_timesteps_per_episode):
            
            # Choosing an action to perform
            ltl_goal = info['ltl_goal']
            
            # check if LTL goal progressed and log results
            if prev_ltl_goal != ltl_goal:
                if do_render: print("    ", t, ":     progressed to", ltl_goal)
                prog_count += 1
                prev_ltl_goal = ltl_goal

            # TODO deterministic evaluation
            a = policy_bank.get_best_action(ltl_goal, np.expand_dims(s1, axis=0), deterministic=deterministic)

            # Executing the action
            s1, r, term, trunc, info = task.step(a)
            r_total += r

            # Restarting the environment (Game Over)
            if info['dfa_game_over'] or trunc or term or t > learning_params.max_timesteps_per_episode:
                # print the final LTL and deadend
                if do_render: print("    ", t, ":     game over. Final LTL:", info['ltl_goal'], "; deadend:", info['dfa_state'] == -1)
                
                if info['ltl_goal'] == "True":
                    succ_count += 1
                    prog_count += 1
                break

            # rendering
            if do_render: task.render()
        if do_render:
            if t == learning_params.max_timesteps_per_episode - 1:
                print("    ", t, ":     reached max time step")
            if do_render:
                time.sleep(0.1)
        r_hist.append(r_total)
        len_hist.append(t + 1)
        prog_count_hist.append(prog_count)

    return {
        "rew_mean": np.mean(r_hist), 
        "rew_std": np.std(r_hist), 
        "len_mean": np.mean(len_hist), 
        "len_std": np.std(len_hist), 
        "succ_rate": succ_count / testing_params.test_epis,
        "prog_count_mean": np.mean(prog_count_hist),
        "prog_count_std": np.std(prog_count_hist)
    }
