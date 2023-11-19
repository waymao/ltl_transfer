import os
import random
import time
import numpy as np
# import tensorflow as tf

from torch_policies.policy_bank import *

from utils.schedules import LinearSchedule
from utils.replay_buffer import ReplayBuffer
from ltl.dfa import *
from envs.game_creator import get_game
from test_utils import Loader, load_pkl

from utils.curriculum import CurriculumLearner
from test_utils import Tester, Saver
from torch.profiler import profile, record_function, ProfilerActivity
import random

def run_experiments(
        game_name: str,
        tester: Tester, 
        curriculum: CurriculumLearner, 
        saver: Saver, run_id: int, 
        num_times, incremental_steps, show_print, 
        rl_algo="dqn", resume=False, device="cpu"):
    time_init = time.time()
    tester_original = tester
    curriculum_original = curriculum
    loader = Loader(saver)
    train_dpath = os.path.join(saver.exp_dir, "train_data")
    
    # seeding
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)

    # Running the tasks 'num_times'
    for run_id in range(num_times):
        run_dpath = os.path.join(train_dpath, "run_%d" % run_id)
        # Overwrite 'tester' and 'curriculum' if incremental training
        tester_fpath = os.path.join(run_dpath, "tester.pkl")
        if resume and os.path.exists(run_dpath) and os.path.exists(tester_fpath):
            tester = load_pkl(tester_fpath)
        else:
            tester = tester_original

        learning_params = tester.learning_params

        curriculum_fpath = os.path.join(run_dpath, "curriculum.pkl")
        if resume and os.path.exists(run_dpath) and os.path.exists(curriculum_fpath):
            curriculum = load_pkl(curriculum_fpath)
            learning_params.learning_starts += curriculum.total_steps  # recollect 'replay_buffer'
            curriculum.incremental_learning(incremental_steps)
        else:
            curriculum = curriculum_original

        # Setting the random seed to 'run_id'
        random.seed(run_id)

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
        try:
            while not curriculum.stop_learning():
                task = curriculum.get_next_task()
                if show_print:
                    print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
                    print("%d Current task: %d, %s" % (num_tasks, curriculum.current_task, str(task)))
                task_params = tester.get_task_params(task)
                _run_LPOPL(game_name, policy_bank, task_params, tester, curriculum, replay_buffer, show_print)
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
    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init)/60), "mins")


def _initialize_policy_bank(game_name, learning_params, curriculum: CurriculumLearner, tester: Tester, load_tf=True, rl_algo="dqn", device="cpu"):
    task_aux = get_game(game_name, tester.get_task_params(curriculum.get_current_task()))
    num_actions = task_aux.action_space.n
    num_features = task_aux.observation_space.shape[0]
    policy_bank = PolicyBank(num_actions, num_features, learning_params, policy_type=rl_algo, device=device)
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
    if load_tf:
        policy_bank.reconnect()  # -> creating the connections between the neural nets
    # print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank


def _run_LPOPL(game_name, policy_bank: PolicyBank, task_params, tester: Tester, curriculum: CurriculumLearner, replay_buffer, show_print):
    MAX_EPS = 1000
    # Initializing parameters
    learning_params: LearningParameters = tester.learning_params
    testing_params = tester.testing_params

    # Initializing the game
    task = get_game(game_name, task_params)
    action_space = task.action_space

    # Initializing parameters
    num_features = task.observation_space.shape[0]
    num_steps = learning_params.max_timesteps_per_task
    exploration = LinearSchedule(schedule_timesteps=int(learning_params.exploration_fraction * num_steps), initial_p=1.0, final_p=learning_params.exploration_final_eps)
    training_reward = 0

    # Starting interaction with the environment
    curr_eps_step = 0
    if show_print: print("Executing", num_steps, "actions...")
    s1 = task.reset()
    s2 = None

    for t in range(num_steps):
        # Getting the current state and ltl goal
        ltl_goal = task.get_LTL_goal()

        # Choosing an action to perform
        if policy_bank.rl_algo == "dqn":
            if random.random() < exploration.value(t): a = action_space.sample()
            else: a = policy_bank.get_best_action(ltl_goal, s1.reshape((1, num_features)))
        else:
            a = policy_bank.get_best_action(ltl_goal, s1.reshape((1, num_features)))
        # updating the curriculum
        curriculum.add_step()

        # Executing the action
        s2, reward, term, trunc, info = task.step(a)
        training_reward += reward
        true_props = task.get_true_propositions()

        # Saving this transition
        next_goals = np.zeros((policy_bank.get_number_LTL_policies(),), dtype=np.float64)
        for ltl in policy_bank.get_LTL_policies():
            ltl_id = policy_bank.get_id(ltl)
            if task.env_game_over:
                ltl_next_id = policy_bank.get_id("False")  # env deadends are equal to achive the 'False' formula
            else:
                ltl_next_id = policy_bank.get_id(policy_bank.get_policy_next_LTL(ltl, true_props))
            next_goals[ltl_id-2] = ltl_next_id
        replay_buffer.add(s1, a, s2, next_goals)

        s1 = s2

        # Learning
        step = curriculum.get_current_step()
        active_policy_metrics = None # for logging loss

        if step > learning_params.learning_starts and step % learning_params.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            S1, A, S2, Goal = replay_buffer.sample(learning_params.batch_size)

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, ) as prof:
            # with profile(activities=[ProfilerActivity.CPU]) as prof:
            #     policy_bank.learn(S1, A, S2, Goal, active_policy=curriculum.current_task)
            # # prof.export_chrome_trace("profile_trace.json")
            # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

            active_policy_metrics = policy_bank.learn(S1, A, S2, Goal, active_policy=curriculum.current_task)
            if step % learning_params.target_network_update_freq == 0:
                # print("step", step, "; loss", loss.cpu().item())
                pass

        # Updating the target network
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.target_network_update_freq == 0:
            # Update target network periodically.
            policy_bank.update_target_network()

        # Printing
        global_step = curriculum.get_current_step() + 1
        if global_step % learning_params.print_freq == 0:
            if show_print:
                print("Step:", curriculum.get_current_step()+1, "\tTotal reward:", training_reward, "\tSucc rate:", "%0.3f"%curriculum.get_succ_rate())
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
            if active_policy_metrics is not None:
                for key, val in active_policy_metrics.items():
                    tester.logger.add_scalar(
                        f"train/active/{key}",
                        val,
                        global_step=curriculum.get_current_step() + 1
                    )

        # Testing
        # if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
        #     tester.run_test(curriculum.get_current_step(), game_name, _test_LPOPL, policy_bank, num_features)

        # Restarting the environment (Game Over)
        curr_eps_step += 1
        if task.ltl_game_over or task.env_game_over or curr_eps_step > learning_params.max_timesteps_per_episode:
            curr_eps_step = 0
            # NOTE: Game over occurs for one of three reasons:
            # 1) DFA reached a terminal state,
            # 2) DFA reached a deadend, or
            # 3) The agent reached an environment deadend (e.g. a PIT)
            # 4) NEW: > episode max time step
            s1 = task.reset()  # Restarting

            # updating the hit rates
            curriculum.update_succ_rate(t, reward)
            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

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


def _test_LPOPL(game_name, task_params, learning_params, testing_params, policy_bank, num_features):
    # Initializing parameters
    task = get_game(game_name, task_params)
    s1 = task.reset()

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Choosing an action to perform
        a = policy_bank.get_best_action(task.get_LTL_goal(), s1.reshape((1, num_features)))

        # Executing the action
        s1, r, term, trunc, info = task.step(a)
        r_total += r * learning_params.gamma**t

        # Restarting the environment (Game Over)
        if task.ltl_game_over or task.env_game_over:
            break
    return r_total
