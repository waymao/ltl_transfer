
# %%
import os
from ltl.dfa import DFA
from utils.learning_params import LearningParameters, add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import TianshouPolicyBank, load_ts_policy_bank
from ts_utils.ts_argparse import add_parser_cmds
# %%
from ts_utils.ts_envs import generate_envs
import numpy as np

from test_utils import TaskLoader, TestingParameters

import time
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
from tianshou.data import Collector
import os

NUM_PARALLEL_JOBS = 20
PARALLEL_TRAIN = True

device = "cpu"


if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    add_parser_cmds(parser)
    parser.add_argument('--run_prefix', type=str,
                        help='Location of the bank and the learning parameters.')

    # add all fields of Learning Parameters to parser
    LEARNING_ARGS_PREFIX = "lp."
    add_fields_to_parser(parser, LearningParameters, prefix=LEARNING_ARGS_PREFIX)

    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output file to save the results.')
    parser.add_argument('--stochastic_eval', action="store_true", help='Whether to run deterministic evaluation or not.')
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

    # path for saves
    testing_params = TestingParameters(custom_metric_folder=args.run_subfolder)
    print("Initialized Learning Params:", learning_params)

    train_envs, test_envs = generate_envs(prob=args.prob,game_name=args.game_name, parallel=PARALLEL_TRAIN, map_id=map_id, seed=args.run_id)

    # tester
    task_loader = TaskLoader(args)
    tasks = task_loader.tasks

    # initalize policy bank
    policy_bank: TianshouPolicyBank = load_ts_policy_bank(
        os.path.join(task_loader.get_save_path()),
        num_actions=test_envs.action_space[0].n,
        num_features=test_envs.observation_space[0].shape[0],
        load_classifier=None
    )
    # sanity check to make sure everything is in the policy bank.
    for task in tasks:
        dfa = DFA(task)
        for ltl in dfa.ltl2state.keys():
            if ltl != 'True' and ltl != 'False' and ltl[0] != 'always': 
                assert ltl in policy_bank.policy2id, \
                    ("LTL " + str(ltl) + " not found in policy bank.")

    # run training
    global_time_steps = 0
    result_list = []
    num_policies = len(policy_bank.policies)
    print("Number of policies:", num_policies)
    print("EVAL mode is stochastic:", args.stochastic_eval)
    for i, (ltl, policy) in enumerate(policy_bank.get_all_policies().items()):
        # skip if it's a dummy policy
        if ltl == "True" or ltl == "False": continue
        
        # reset with the correct ltl
        task_params = task_loader.get_task_params(ltl)
        test_envs.reset(options=dict(task_params=task_params))
        
        # collecting results
        # set policy to be deterministic if set up to do so
        if not args.stochastic_eval:
            # policy.training = False
            policy.eval()

        # collect and rollout
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        result = test_collector.collect(n_episode=20)

        print(f"LTL {i}/{num_policies}:", ltl, \
              "rew:", f"{result['rew']} ± {result['rew_std']}", \
              "len:", f"{result['len']} ± {result['len_std']}")
        result_list.append((ltl, result['rew'], result['rew_std'], result['len'], result['len_std']))
    
    print("Average Success Rate:", np.mean([rew for _, rew, _, _, _ in result_list]) / 10)

    # save results
    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            f.write("ltl,rew,rew_std,length,length_std\n")
            for ltl, rew, rew_std, length, length_std in result_list:
                f.write(f"\"{ltl}\",{rew},{rew_std},{length},{length_std}\n")

