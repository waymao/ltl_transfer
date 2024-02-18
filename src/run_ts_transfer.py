import cProfile
from copy import deepcopy
import gzip
import json
import os

import gymnasium
import networkx as nx
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters
from ts_utils.matcher import dfa2graph, get_training_edges, match_remove_edges
from ts_utils.policy_switcher import PolicySwitcher

from ts_utils.ts_policy_bank import load_ts_policy_bank
from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

from utils.print_ltl import ltl_to_print

# %%
from tianshou.data import  Batch
import numpy as np

from test_utils import TaskLoader, TestingParameters

import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
import os

import time


NUM_PARALLEL_JOBS = 10
PARALLEL_TRAIN = False

device = "cpu"



def run_experiment():
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    add_parser_cmds(parser)

    # add all fields of Learning Parameters to parser
    LEARNING_ARGS_PREFIX = "lp."
    add_fields_to_parser(parser, LearningParameters, prefix=LEARNING_ARGS_PREFIX)
    parser.add_argument('--run_prefix', type=str,
                        help='Location of the bank and the learning parameters.')
    parser.add_argument('--task_id', type=int, help='The task id to run.')
    parser.add_argument('--no_deterministic_eval', action="store_true", help='Whether to run deterministic evaluation or not.')
    parser.add_argument('--render', action="store_true", help='Whether to run rendering.')
    parser.add_argument('--verbose', '-v', action="store_true", help='Whether to print debug info.')
    parser.add_argument('--num_epi', default=100, type=int, help="Number of Episodes to Run.")

    args = parser.parse_args()
    # if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    # if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    # if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    # if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = 0
    map_id = args.map

    # learning params
    learning_params = get_learning_parameters(
        policy_name=args.rl_algo, 
        game_name=args.game_name,
        **{
            key.removeprefix(LEARNING_ARGS_PREFIX): val 
            for (key, val) in args._get_kwargs() 
            if key.startswith(LEARNING_ARGS_PREFIX)
        }
    )
    testing_params = TestingParameters(custom_metric_folder=args.run_subfolder)
    print("Initialized Learning Params:", learning_params)

    train_envs, test_envs = generate_envs(prob=args.prob,
        game_name=args.game_name, 
        parallel=PARALLEL_TRAIN, 
        map_id=map_id, 
        seed=args.run_id,
        no_info=False,
        ltl_progress_is_term=False,
        max_episode_steps=9000
    )
    
    

    # tester
    task_loader = TaskLoader(args)
    tasks = task_loader.transfer_tasks
    
    # load the proper lp
    with open(os.path.join(task_loader.get_save_path(), "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)

    # sampler
    env_size = test_envs.get_env_attr("size", 0)[0]
    state_space = gymnasium.spaces.Box(
        low=np.array([1, 1, -180]), 
        high=np.array([env_size - 1, env_size - 1, 180])
    )

    # run training
    policy_bank = load_ts_policy_bank(
        task_loader.get_save_path(), 
        num_actions=test_envs.action_space[0].n,
        num_features=test_envs.observation_space[0].shape[0],
        hidden_layers=[256, 256, 256],
        learning_params=learning_params,
        device=device,
        verbose=True
    )
    tasks = task_loader.get_LTL_tasks()
    try:
        ltl = tasks[args.task_id]
    except IndexError:
        print("Task ID", args.task_id, "not found in the task list.")
        exit(1)

    print("Running task", ltl_to_print(ltl))
    
    # reset with the correct ltl
    task_params = task_loader.get_task_params(ltl)
    obs, info = test_envs.reset(options=dict(task_params=task_params))
    env_state = test_envs.get_env_attr("curr_state", 0)[0]

    # set policy to be deterministic if set up to do so
    if args.no_deterministic_eval:
        raise NotImplementedError("Deterministic evaluation not implemented.")

    # collect and rollout
    results = {}
    if args.render:
        test_envs.render()
        input("Press Enter When Ready...")

    # edge matching
    if args.verbose: print("Gathering training edges from the bank...")
    train_edges, t_edge2ltls = get_training_edges(policy_bank)
    task_dfa: DFA = deepcopy(test_envs.get_env_attr("dfa")[0])
    dfa_graph = dfa2graph(test_envs.get_env_attr("dfa")[0])

    # look for infeasible edges in the testing ("eval") DFA and remove it
    if args.verbose: print("Matching training/testing edges and removing infeasible edges...")
    begin_time = time.time()
    test2trains = match_remove_edges(
        dfa_graph, train_edges, task_dfa.state, task_dfa.terminal[0], task_loader.edge_matcher
    )
    policy_switcher = PolicySwitcher(policy_bank, test2trains, t_edge2ltls, ltl)
    print("Time taken to match training and testing edges:", (time.time() - begin_time), "seconds.")

    # TODO save some metrics
    run_info = {
        "task": tasks[args.task_id],
        "result": []
    }
    if args.verbose: print("Running the experiment...")
    succ_count = 0
    for epi in range(args.num_epi):
        if args.verbose: print("Episode", epi)
        # reset
        policy_switcher.reset_excluded_policy()
        obs, info = test_envs.reset()
        if args.render:
            test_envs.render()
        init_env_state = info[0]['loc']
        
        # rollout of one episode
        node2option2prob = {} # node -> (ltl, edge_pair) -> prob
        term = trunc = [False]
        i1 = 0 # global step counter
        cum_rew = 0

        FAIL_STATUS = ""

        while not term[0] and not trunc[0]:
            # gather the informations
            env_state = info[0]['loc']
            curr_node = info[0]['dfa_state']
            next_node = curr_node

            while next_node == curr_node and not term[0] and not trunc[0]:
                # try every possible option
                env_state = info[0]['loc']
                best_policy, training_edges, ltl, stats = policy_switcher.get_best_policy(curr_node, env_state)
                if best_policy is None:
                    FAIL_STATUS = f"No policy available for goal {ltl_to_print(info[0]['ltl_goal'])}"
                    if args.verbose: print("No policy available / works for node", curr_node, "; goal: ", ltl_to_print(info[0]['ltl_goal']))
                    break

                if args.verbose: 
                    print("Executing policy", ltl_to_print(ltl), 
                          "with training edges", training_edges,
                          "on node", curr_node,
                          "with env state", env_state
                    )
                    print("       test stats: prob:", stats[0], "| len:", stats[1])

                for _ in range(500): # option step limit
                    a = best_policy.forward(Batch(obs=obs, info=info)).act
                    obs, reward, term, trunc, info = test_envs.step(a.numpy())

                    next_node = info[0]['dfa_state']
                    i1 += 1
                    cum_rew += reward

                    if args.render: test_envs.render()
                    if next_node != curr_node and args.verbose: 
                        print("Hit proposition:", info[0]['true_props'], "in", _, "steps.")
                    
                    if term[0] or trunc[0] or next_node != curr_node:
                        break
                
                if not term[0] and not trunc[0] and next_node == curr_node:
                    # we are stuck in the same node
                    policy_switcher.exclude_policy(curr_node, best_policy)
                    if args.verbose: print("   Policy failed to finish. Excluding policy", ltl_to_print(ltl), "on node", curr_node)

            if trunc[0] or term[0] or FAIL_STATUS != "": # env game over
                success = bool(term[0] and info[0]['dfa_state'] != -1 and not trunc[0])
                env_state = test_envs.get_env_attr("curr_state", 0)[0]
                if info[0]['dfa_state'] == -1:
                    FAIL_STATUS = "DFA Dead end"
                elif trunc:
                    FAIL_STATUS = "Truncated by ENV"
                elif term and not success:
                    FAIL_STATUS = "ENV Dead end"
                elif FAIL_STATUS == "" and info[0]['dfa_state'] != task_dfa.terminal[0]:
                    FAIL_STATUS = "DFA not terminal"
                result = {
                    "success": success,
                    "steps": i1 + 1,
                    "message": "success" if success else FAIL_STATUS,
                    "final_env_state": env_state, 
                    "final_dfa_state": info[0]['dfa_state'],
                    "init_env_state": init_env_state
                }
                if success: succ_count += 1
                print(result)
                run_info["result"].append(result)
                break
    print("Done! Success Rate:", succ_count / args.num_epi)
    run_info['success_rate'] = succ_count / args.num_epi

    save_folder = os.path.join(task_loader.get_save_path(), "transfer_results")
    os.makedirs(save_folder, exist_ok=True)
    file_name = os.path.join(save_folder, f"task_{task_loader.test_type}_{args.task_id}.json.gz")
    with gzip.open(file_name, 'wt') as f:
        json.dump(run_info, f)
    print("Saved result to", file_name)


if __name__ == "__main__":
    run_experiment()
    # cProfile.run("run_experiment()", "transfer_prof.prof")

# %%
