from algos.lpopl import _initialize_policy_bank, _test_LPOPL
import argparse
from envs.game_creator import get_game
from test_utils import Tester

from torch_policies.learning_params import LearningParameters, add_fields_to_parser, get_learning_parameters
from test_utils import TestingParameters
from utils.curriculum import CurriculumLearner

import torch
import numpy
import random


if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    # Getting params
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "lpopl_ppo", "lpopl_dsac", "zero_shot_transfer", "random_transfer"]
    rl_algos = ["dqn", "dsac"]
    train_types = [
        "sequence",
        "test_until",
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
    relabel_methods = ["cluster", "local"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    parser.add_argument('--algo', default='lpopl', type=str,
                        help='This parameter indicated which algorithm to use. The options are: ' + str(algos))
    parser.add_argument('--rl_algo', default="dqn", type=str, choices=rl_algos,
                        help="The RL Algorithm to be used for LPOPL / Transfer.  The options are: " + str(rl_algos))
    
    parser.add_argument('--train_type', default='sequence', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=50, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='mixed', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between -2 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map. Use "-2" to generate a random map.s')
    parser.add_argument('--prob', default=1.0, type=float,
                        help='probability of intended action succeeding')
    parser.add_argument('--total_steps', default=500000, type=int,
                        help='This parameter indicated the total training steps to learn all tasks')
    parser.add_argument('--incremental_steps', default=150000, type=int,
                        help='This parameter indicated the increment to the total training steps for additional training')
    parser.add_argument('--run_id', default=0, type=int,
                        help='This parameter indicated the policy bank saved after which run will be used for transfer')
    # parser.add_argument('--load_trained', action="store_true",
    #                     help='This parameter indicated whether to load trained policy models. Include it in command line to load trained policies')
    parser.add_argument('--relabel_method', default='cluster', type=str, choices=["cluster", "local"],
                        help='This parameter indicated which method is used to relabel state-centric options')
    parser.add_argument('--transfer_num_times', default=1, type=int,
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--edge_matcher', default='relaxed', type=str, choices=['rigid', 'relaxed'],
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--save_dpath', default='..', type=str,
                        help='path to directory to save options and results')
    parser.add_argument('--domain_name', '--dataset_name', default='minecraft', type=str, choices=['minecraft', 'spot', 'spot_1', 'spot_mixed20-mixed50'],
                        help='This parameter indicated the dataset to read tasks from')
    parser.add_argument('--device', default="cpu", type=str, choices=['cpu', 'cuda'], 
                        help='The device to run Neural Network computations.')
    # parser.add_argument('--alpha', default=0.1, type=float,
    #                     help='The temperature for exploration / exploitation tradeoff.')
    # parser.add_argument('--auto_alpha', default=False, action="store_true",
    #                     help='Whether to auto tune alpha based on entropy.')
    parser.add_argument('--resume', default=False, action="store_true",
                        help='Whether to resume from a checkpoint or not.')
    parser.add_argument('--game_name', default="grid", type=str, choices=['grid', 'miniworld', 'miniworld_no_vis', 'miniworld_simp_no_vis'],
                        help='Name of the game.')
    parser.add_argument('--run_subfolder', default=None, required=False, type=str,
                        help='Name of the run. Used to save the results in a separate sub folder.')
    parser.add_argument('--run_prefix', type=str,
                        help='Location of the bank and the learning parameters.')

    # add all fields of Learning Parameters to parser
    LEARNING_ARGS_PREFIX = "lp."
    add_fields_to_parser(parser, LearningParameters, prefix=LEARNING_ARGS_PREFIX)
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = train_types.index(args.train_type)
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
    print("Initialized Learning Params:", learning_params)

    # Setting the experiment
    testing_params = TestingParameters(test_epis=100)
    print(learning_params)
    

    tester = Tester(learning_params, testing_params, map_id, args.prob, \
                    tasks_id, args.domain_name, args.train_type, args.train_size, args.test_type, args.edge_matcher, 
                    args.rl_algo, args.save_dpath, None)
    curriculum = CurriculumLearner(tester.tasks)
    ltl_task = tester.tasks[0]
    # task = get_game(args.game_name, tester.get_task_params(curriculum.get_current_task()))
    task = get_game(args.game_name, tester.get_task_params(ltl_task))
    task.reset(seed=args.run_id)

    policy_bank = _initialize_policy_bank(
        args.game_name, learning_params, 
        curriculum, tester, 
        rl_algo=args.rl_algo, device=args.device)
    policy_bank.load_bank(args.run_prefix)

    torch.manual_seed(1)
    numpy.random.seed(1)
    random.seed(1)
    task.render()
    input("Press Enter to begin...")
    result = _test_LPOPL(task, learning_params, testing_params, policy_bank, do_render=True)
    print(result)
