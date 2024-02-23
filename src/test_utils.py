import os
import json
import dill
import argparse
import logging
from collections import defaultdict
import numpy as np
from envs.grid.game import GameParams as GridGameParams, Game
from envs.miniworld.params import GameParams as MiniWorldGameParams
from exp_dataset_creator import read_train_test_formulas
from torch.utils.tensorboard import SummaryWriter
import ltl.tasks as tasks


class TestingParameters:
    def __init__(self, test=True, test_freq=1000, num_steps=1000, test_epis=20, test_seed=5, custom_metric_folder=None):
        """Parameters
        -------
        test: bool
            if True, we test current policy during training
        test_freq: int
            test the model every `test_freq` steps.
        num_steps: int
            number of steps during testing
        """
        self.test = test
        self.test_freq = test_freq
        self.num_steps = num_steps
        self.test_epis = test_epis
        self.test_seed = test_seed
        self.test_env_instances = 8
        self.custom_metric_folder = custom_metric_folder


class TaskLoader:
    def __init__(self, 
                 args, create_logger=False, render=False
        ):
        # setting the test attributes
        self.map_id = args.map
        self.transition_type = "deterministic" if args.prob == 1.0 else "stochastic"
        self.prob = args.prob
        self.dataset_name = args.domain_name
        self.train_type = args.train_type
        self.train_size = args.train_size
        self.test_type = args.test_type
        self.edge_matcher = args.edge_matcher
        self.save_dpath = args.save_dpath
        self.experiment = f"{self.train_type}/map_{self.map_id}"
        self.game_name = args.game_name
        if self.map_id == -2:
            self.map = None
        else:
            self.map = f"{self.save_dpath}/experiments/maps/map_{self.map_id}.txt"
        self.consider_night = False
        self.rl_algo = args.rl_algo

        results_path = os.path.join(self.save_dpath, "saves")

        # get the base run path
        self.log_path = os.path.join(
            results_path,
            f"{args.game_name}_{args.domain_name}_p{args.prob}", 
            f"{args.train_type}_{args.train_size}", 
            f"{args.rl_algo}", 
            f"map{self.map_id}", str(args.run_id), 
        )
        if args.run_subfolder is not None:
            self.log_path = os.path.join(self.log_path, args.run_subfolder)

        # logger
        if create_logger:
            from tianshou.utils.logger.tensorboard import TensorboardLogger
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_path, "logs", f"policy_{args.ltl_id}"))
            self.logger = TensorboardLogger(self.writer)
        else:
            self.logger = None
            self.writer = None

        # loading tasks
        if self.train_type == "sequence":
            # original LPOPL code, testing use only
            self.tasks = tasks.get_sequence_of_subtasks()
        elif self.train_type == "test_until":
            # testing use only
            self.tasks = tasks.get_sequence_of_until()
        # elif train_type == "interleaving":
        #     self.tasks = tasks.get_interleaving_subtasks()
        # elif train_type == "safety":
        #     self.tasks = tasks.get_safety_constraints()
        #     self.consider_night = True
        elif self.train_type == 'random':
            train_tasks, self.transfer_tasks = read_train_test_formulas(self.dataset_name, 'hard', self.test_type, self.train_size)
            self.tasks = train_tasks[0: self.train_size]
        else:  # transfer tasks
            if self.train_type == 'transfer_sequence':
                self.tasks = tasks.get_sequence_training_tasks()
                self.transfer_tasks = tasks.get_transfer_tasks()
                self.transfer_results_dpath = os.path.join(results_path, self.train_type, f"map_{self.map_id}", f"prob_{self.prob}")
            elif self.train_type == 'transfer_interleaving':
                self.tasks = tasks.get_interleaving_training_tasks()
                self.transfer_tasks = tasks.get_transfer_tasks()
                self.transfer_results_dpath = os.path.join(results_path, self.train_type, f"map_{self.map_id}", f"prob_{self.prob}")
            else:
                self.experiment = f"{self.train_type}_{self.train_size}/map_{self.map_id}/prob_{self.prob}"
                self.experiment_train = f"{self.train_type}_50/map_{self.map_id}/prob_{self.prob}"
                train_tasks, self.transfer_tasks = read_train_test_formulas(
                    self.save_dpath, self.dataset_name, self.train_type, self.test_type, self.train_size)
                self.tasks = train_tasks[0: self.train_size]

        # load pre-computed optimal steps for 'task_type' in 'map_id'
        # optimal_aux = _get_optimal_values(f'{save_dpath}/experiments/optimal_policies/map_{map_id}.txt', tasks_id)
    
    def get_save_path(self):
        return self.log_path

    def get_LTL_tasks(self):
        return self.tasks

    def get_transfer_tasks(self):
        return self.transfer_tasks

    def get_task_params(self, ltl_task, init_dfa_state=None, init_loc=None):
        if self.game_name == "grid":
            return GridGameParams(self.map, self.prob, ltl_task, self.consider_night, init_dfa_state, init_loc)
        else:
            return MiniWorldGameParams(self.map, self.prob, ltl_task, self.consider_night, init_dfa_state, init_loc)


def save_pkl(fpath, data):
    with open(fpath, "wb") as file:
        dill.dump(data, file)


def load_pkl(fpath):
    with open(fpath, 'rb') as file:
        data = dill.load(file)
    return data


def save_json(fpath, data):
    with open(fpath, 'w') as outfile:
        json.dump(data, outfile)


def read_json(fpath):
    with open(fpath) as data_file:
        data = json.load(data_file)
    return data


if __name__ == "__main__":
    # EXAMPLE for export training results: python test_utils.py --algo=lpopl --train_type=sequence
    # EXAMPLE for export transfer results: python test_utils.py --train_type=no_orders --train_size=50 --test_type=hard --map=0 --transfer_num_times=1

    # Getting params
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "zero_shot_transfer"]
    train_types = [
        "sequence",
        "interleaving",
        "safety",
        "transfer_sequence",
        "transfer_interleaving",
        "hard",
        "mixed",
        "soft_strict",
        "soft",
        "no_orders",
    ]
    test_types = [
        "hard",
        "mixed",
        "soft_strict",
        "soft",
        "no_orders",
    ]

    parser = argparse.ArgumentParser(prog="run_experiments", description="Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.")
    parser.add_argument("--algo", default="lpopl", type=str,
                        help="This parameter indicated which RL algorithm to use. The options are: " + str(algos))
    parser.add_argument('--train_type', default='no_orders', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=50, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='soft', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--prob', default=1.0, type=float,
                        help='transition stochasticity')
    parser.add_argument('--transfer_num_times', default=1, type=int,
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--edge_matcher', default='rigid', type=str, choices=['rigid', 'relaxed'],
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--save_dpath', default='..', type=str,
                        help='path to directory to save')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError(
        "Training tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError(
        "Test tasks " + str(args.test_type) + " hasn't been defined yet")
    if not (-1 <= args.map < 10): raise NotImplementedError("The map must be a number between -1 and 9")
