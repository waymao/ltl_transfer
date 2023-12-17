import argparse
# from algos import baseline_dqn
# from algos import baseline_hrl
from algos import lpopl
from algos import transfer
# from algos import random_transfer
from test_utils import TestingParameters, Tester, Saver
from utils.curriculum import CurriculumLearner
from torch_policies.learning_params import LearningParameters, get_learning_parameters
from torch.utils.tensorboard import SummaryWriter
import os
import cProfile



if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    # Getting params
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "lpopl_ppo", "lpopl_dsac", "zero_shot_transfer", "random_transfer"]
    rl_algos = ["dqn", "dsac"]
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
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='The temperature for exploration / exploitation tradeoff.')
    parser.add_argument('--auto_alpha', default=False, action="store_true",
                        help='Whether to auto tune alpha based on entropy.')
    parser.add_argument('--resume', default=False, action="store_true",
                        help='Whether to resume from a checkpoint or not.')
    parser.add_argument('--game_name', default="grid", type=str, choices=['grid', 'miniworld', 'miniworld_no_vis'],
                        help='Name of the game.')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    