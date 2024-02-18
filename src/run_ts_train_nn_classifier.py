import json
import os
import sys

from classifiers.nn_classifier import NNClassifier
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

from test_utils import TaskLoader, TestingParameters

import argparse


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
    parser.add_argument("--resume_from", default=0, type=int, help="Resume from a specific policy ID.")
    parser.add_argument("--rollout_method", default="random", type=str, help="Resume from a specific policy ID.")

    args = parser.parse_args()


    # tester
    task_loader = TaskLoader(args)
    tasks = task_loader.tasks

    # run training
    with open(os.path.join(task_loader.get_save_path(), "ltl_list.json"), "r") as f:
        ltl_list = json.load(f)

    for i in range(len(ltl_list)):
        if i < args.resume_from:
            continue
        print(f"Running training for {i}")
        classifier = NNClassifier()
        try:
            classifier.load_raw_data(task_loader.get_save_path(), i, args.rollout_method)
            classifier.train(verbose=False)
            classifier.save(task_loader.get_save_path(), i)
        except Exception as e:
            print(f"Failed to train classifier for {i}: {e}", file=sys.stderr)
            continue
