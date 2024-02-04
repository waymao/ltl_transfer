from copy import deepcopy
import json
import os

import gymnasium
import networkx as nx
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters
from ts_utils.matcher import dfa2graph, get_training_edges, match_remove_edges
from ts_utils.policy_switcher import PolicySwitcher

from ts_utils.ts_policy_bank import TianshouPolicyBank, load_ts_policy_bank
from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

# %%
from tianshou.data import  Batch
import numpy as np

from test_utils import Tester, TestingParameters

import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
import os


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
    parser.add_argument('--no_deterministic_eval', action="store_true", help='Whether to run deterministic evaluation or not.')
    parser.add_argument('--render', action="store_true", help='Whether to run rendering.')

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

    train_envs, test_envs = generate_envs(
        game_name=args.game_name, 
        parallel=PARALLEL_TRAIN, 
        map_id=map_id, 
        seed=args.run_id,
        no_info=False,
        ltl_progress_is_term=False
    )

    # path for logger
    tb_log_path = os.path.join(
        args.save_dpath, "results", f"{args.game_name}_{args.domain_name}", f"{args.train_type}_p{args.prob}", 
        f"{args.algo}_{args.rl_algo}", f"map{map_id}", str(args.run_id), 
        f"alpha={'auto' if learning_params.auto_alpha else learning_params.alpha}",
    ) if args.run_prefix is None else args.run_prefix
    if testing_params.custom_metric_folder is not None:
        tb_log_path = os.path.join(tb_log_path, testing_params.custom_metric_folder)
    
    # load the proper lp
    with open(os.path.join(tb_log_path, "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)

    # tester
    tester = Tester(
        learning_params=learning_params, 
        testing_params=testing_params,
        map_id=args.map,
        prob=map_id,
        train_size=args.train_size,
        rl_algo=args.rl_algo,
        tasks_id=tasks_id,
        dataset_name=args.domain_name,
        train_type=args.train_type,
        test_type=args.test_type,
        edge_matcher=args.edge_matcher, 
        save_dpath=args.save_dpath,
        game_name=args.game_name,
        logger=None
    )
    tasks = tester.tasks

    # sampler
    env_size = test_envs.get_env_attr("size", 0)[0]
    state_space = gymnasium.spaces.Box(
        low=np.array([1, 1, -180]), 
        high=np.array([env_size - 1, env_size - 1, 180])
    )

    # run training
    policy_bank = load_ts_policy_bank(
        tb_log_path, 
        num_actions=test_envs.action_space[0].n,
        num_features=test_envs.observation_space[0].shape[0],
        hidden_layers=[256, 256, 256],
        learning_params=learning_params,
        device=device,
        verbose=True
    )
    tasks = tester.get_LTL_tasks()
    ltl = tasks[0]

    print("Running task", ltl)
    
    # reset with the correct ltl
    task_params = tester.get_task_params(ltl)
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
    train_edges, t_edge2ltls = get_training_edges(policy_bank)
    task_dfa: DFA = deepcopy(test_envs.get_env_attr("dfa")[0])
    dfa_graph = dfa2graph(test_envs.get_env_attr("dfa")[0])

    # look for infeasible edges in the testing ("eval") DFA and remove it
    test2trains = match_remove_edges(
        dfa_graph, train_edges, task_dfa.state, task_dfa.terminal[0], tester.edge_matcher
    )
    policy_switcher = PolicySwitcher(policy_bank, test2trains, t_edge2ltls, ltl)

    # TODO save some metrics
    for epi in range(100):
        # reset
        policy_switcher.reset_excluded_policy()
        obses, infos = test_envs.reset()
        if args.render:
            test_envs.render()
        
        # rollout of one episode
        node2option2prob = {} # node -> (ltl, edge_pair) -> prob
        term = trunc = False
        i1 = 0 # global step counter
        cum_rew = 0

        while not term and not trunc:
            # gather the informations
            env_state = infos[0]['loc']
            curr_node = infos[0]['dfa_state']
            next_node = curr_node

            while next_node == curr_node and not term and not trunc:
                # try every possible option
                training_edges, best_policy = policy_switcher.get_best_policy(curr_node, env_state)

                for _ in range(500): # option step limit
                    a = best_policy.forward(Batch(obs=obs, info=info)).act
                    obs, reward, term, trunc, info = test_envs.step(a.numpy())

                    next_node = info[0]['dfa_state']
                    i1 += 1
                    cum_rew += reward

                    if args.render: test_envs.render()
                    if term or trunc or next_node != curr_node:
                        break
                
                if not term and not trunc and next_node == curr_node:
                    # we are stuck in the same node
                    policy_switcher.exclude_policy(training_edges, best_policy)

            if trunc or term: # env game over
                success = info[0]['dfa_state'] != -1 and not trunc
                env_state = test_envs.get_env_attr("curr_state", 0)[0]
                result = {
                    "success": success,
                    "steps": i1 + 1,
                    "final_state": env_state, 
                }
                print(result)
                break

        print("Done!")


if __name__ == "__main__":
    run_experiment()

# %%
