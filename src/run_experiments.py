import argparse
# from algos import baseline_dqn
# from algos import baseline_hrl
from algos import lpopl
from algos import lpopl_online
from algos import transfer
# from algos import random_transfer
from test_utils import TestingParameters, Tester, Saver
from utils.curriculum import CurriculumLearner
from torch_policies.learning_params import LearningParameters
import cProfile



def run_experiment(
        alg_name, map_id, prob, tasks_id, dataset_name, train_type, 
        train_size, test_type, num_times, r_good, total_steps, incremental_steps, 
        run_id, relabel_method, transfer_num_times, edge_matcher, save_dpath, show_print,
        device):
    # configuration of learning params
    learning_params = LearningParameters()

    # configuration of testing params
    testing_params = TestingParameters()

    # Setting the experiment
    tester = Tester(learning_params, testing_params, map_id, prob, tasks_id, dataset_name, train_type, train_size, test_type, edge_matcher, save_dpath)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.tasks, r_good=r_good, total_steps=total_steps)

    # Setting up the saver
    saver = Saver(alg_name, tester)

    # # Baseline 1 (standard DQN with Michael Littman's approach)
    # if alg_name == "dqn-l":
    #     baseline_dqn.run_experiments(tester, curriculum, saver, num_times, show_print)

    # # Baseline 2 (Hierarchical RL)
    # if alg_name == "hrl-e":
    #     baseline_hrl.run_experiments(tester, curriculum, saver, num_times, show_print, use_dfa=False)

    # # Baseline 3 (Hierarchical RL with LTL constraints)
    # if alg_name == "hrl-l":
    #     baseline_hrl.run_experiments(tester, curriculum, saver, num_times, show_print, use_dfa=True)

    # LPOPL
    if alg_name == "lpopl":
        lpopl.run_experiments(tester, curriculum, saver, num_times, incremental_steps, show_print, device=device)
    if alg_name == "lpopl_dsac":
        lpopl.run_experiments(tester, curriculum, saver, num_times, incremental_steps, show_print, rl_algo="dsac", device=device)
    if alg_name == "lpopl_ppo":
        lpopl_online.run_experiments(tester, curriculum, saver, num_times, incremental_steps, show_print, device=device)

    # # Relabel state-centric options learn by LPOPL then zero-shot transfer
    if alg_name == "zero_shot_transfer":
        transfer.run_experiments(tester, curriculum, saver, run_id, relabel_method, transfer_num_times)

    # # Random policy baseline
    # if alg_name == "random_transfer":
    #     random_transfer.run_experiments(tester, curriculum, saver, run_id, relabel_method, transfer_num_times)


def run_multiple_experiments(alg, prob, tasks_id, dataset_name, train_type, train_size, test_type, total_steps, incremental_steps, run_id, relabel_method, transfer_num_times, edge_matcher, save_dpath, device):
    num_times = 3
    r_good    = 0.5 if tasks_id == 2 else 0.9
    show_print = True

    for map_id in range(10):
        print("Running r_good: %0.2f; alg: %s; map_id: %d; stochasticity: %0.2f; train_type: %s; train_size: %d; test_type: %s; edge_mather: %s" % (r_good, alg, map_id, prob, train_type, train_size, test_type, edge_matcher))
        run_experiment(alg, map_id, prob, tasks_id, dataset_name, train_type, 
                       train_size, test_type, num_times, r_good, total_steps, incremental_steps, 
                       run_id, relabel_method, transfer_num_times, edge_matcher, save_dpath, show_print, device)


def run_single_experiment(alg, map_id, prob, tasks_id, dataset_name, train_type, train_size, test_type, total_steps, incremental_steps, run_id, relabel_method, transfer_num_times, edge_matcher, save_dpath, device):
    num_times = 1  # each algo was run 3 times per map in the paper
    r_good    = 0.5 if tasks_id == 2 else 0.9
    show_print = True

    print("Running r_good: %0.2f; alg: %s; map_id: %d; stochasticity: %0.2f; train_type: %s; train_size: %d; test_type: %s; edge_mather: %s" % (r_good, alg, map_id, prob, train_type, train_size, test_type, edge_matcher))
    run_experiment(alg, map_id, prob, tasks_id, dataset_name, train_type, train_size, test_type, 
                   num_times, r_good, total_steps, incremental_steps, run_id, relabel_method, 
                   transfer_num_times, edge_matcher, save_dpath, show_print, device)


if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    # Getting params
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "lpopl_ppo", "lpopl_dsac", "zero_shot_transfer", "random_transfer"]
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
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algos))
    parser.add_argument('--train_type', default='sequence', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=50, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='mixed', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
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
    parser.add_argument('--dataset_name', default='minecraft', type=str, choices=['minecraft', 'spot', 'spot_1', 'spot_mixed20-mixed50'],
                        help='This parameter indicated the dataset to read tasks from')
    parser.add_argument('--device', default="cpu", type=str, choices=['cpu', 'cuda'], 
                        help='The device to run Neural Network computations.')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    if not(-1 <= args.map < 10 or args. map == 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = train_types.index(args.train_type)
    map_id = args.map
    if map_id > -1:
        run_single_experiment(args.algo, map_id, args.prob, tasks_id, args.dataset_name, args.train_type, args.train_size, args.test_type,
                              args.total_steps, args.incremental_steps, args.run_id,
                              args.relabel_method, args.transfer_num_times, args.edge_matcher, args.save_dpath, args.device)
    else:
        run_multiple_experiments(args.algo, args.prob, tasks_id, args.dataset_name, args.train_type, args.train_size, args.test_type,
                                 args.total_steps, args.incremental_steps, args.run_id,
                                 args.relabel_method, args.transfer_num_times, args.edge_matcher, args.save_dpath, args.device)
