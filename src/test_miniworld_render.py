
# %%
import os
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, add_fields_to_parser

from ts_utils.ts_argparse import add_parser_cmds
# %%
from ts_utils.ts_envs import generate_envs

from test_utils import TaskLoader, TestingParameters

import argparse
import os

NUM_PARALLEL_JOBS = 12
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

    args = parser.parse_args()
    # if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    # if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    # if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    # if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = 0
    map_id = args.map

    testing_params = TestingParameters(custom_metric_folder=args.run_subfolder)
    train_envs, test_envs = generate_envs(prob=args.prob,
        game_name=args.game_name, parallel=False, no_info=False, map_id=map_id, seed=args.run_id)

    obs = test_envs.reset()
    test_envs.render()
    for i in range(2000):
        a = input("action: ")
        try:
            a = int(a)
        except:
            print("Invalid action")
            continue
        result = test_envs.step([int(a)])
        print(result[1:])
        print(result[0].reshape(-1, 8))
        test_envs.render()
