
# %%
import os
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import \
    create_discrete_sac_policy, TianshouPolicyBank, load_ts_policy_bank, \
    create_vis_discrete_sac_policy
from ts_utils.ts_argparse import add_parser_cmds
# %%
from ts_utils.ts_envs import generate_envs
import torch
from torch import nn
import numpy as np

from test_utils import TaskLoader, TestingParameters

import time
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
import os

NUM_PARALLEL_JOBS = 10
PARALLEL_TRAIN = False

device = "cpu"


if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    add_parser_cmds(parser)

    # add all fields of Learning Parameters to parser
    LEARNING_ARGS_PREFIX = "lp."
    add_fields_to_parser(parser, LearningParameters, prefix=LEARNING_ARGS_PREFIX)

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

    train_envs, test_envs = generate_envs(prob=args.prob,game_name=args.game_name, parallel=PARALLEL_TRAIN, map_id=map_id, seed=args.run_id)

    task_loader = TaskLoader(args)

    os.makedirs(task_loader.get_save_path(), exist_ok=True)
    with open(os.path.join(task_loader.get_save_path(), "learning_params.pkl"), "wb") as f:
        pickle.dump(learning_params, f)

    # tester
    tester = TaskLoader(args)
    tasks = tester.tasks

    # initalize policy bank
    policy_bank = TianshouPolicyBank()

    for task in tasks:
        dfa = DFA(task)
        for ltl in dfa.ltl2state.keys():
            if ltl[0] == 'always': continue # skip always statements as they are treated as eventual
            
            if len(test_envs.observation_space[0].shape) == 1:
                policy = create_discrete_sac_policy(
                    num_actions=test_envs.action_space[0].n, 
                    num_features=test_envs.observation_space[0].shape[0], 
                    hidden_layers=[256, 256, 256],
                    learning_params=learning_params,
                    device=device
                )
            else:
                policy = create_vis_discrete_sac_policy(
                    num_actions=test_envs.action_space[0].n, 
                    input_size=test_envs.observation_space[0].shape, 
                    hidden_layers=[256, 256, 256],
                    learning_params=learning_params,
                    device=device
                )
            policy_bank.add_LTL_policy(ltl, policy)

    # save policy bank
    print("Created bank with {} policies.".format(len(policy_bank.policies)))
    policy_bank.save_pb_index(task_loader.get_save_path())
    policy_bank.save_ckpt(task_loader.get_save_path())
    print("Saved bank to", task_loader.get_save_path())


    # sanity check
    # load policy bank
    print("Running Sanity Check, Loading Policy Bank.")
    policy_bank = load_ts_policy_bank(
        task_loader.get_save_path(),
        num_actions=test_envs.action_space[0].n,
        num_features=test_envs.observation_space[0].shape[0],
        learning_params=learning_params,
        load_classifier=None,
        device=device
    )
    print("Successfully loaded policy bank. Sanity Check Complete.")
    print("Policy Bank saved at:", task_loader.get_save_path())

