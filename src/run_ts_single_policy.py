
# %%
import os
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import create_discrete_sac_policy, TianshouPolicyBank, load_ts_policy_bank, load_individual_policy, save_individual_policy
from ts_utils.ts_envs import generate_envs

# %%
from tianshou.policy import PPOPolicy, DiscreteSACPolicy, TD3Policy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from torch_policies.network import get_CNN_preprocess
from torch.optim import Adam
import torch
from torch import nn
import numpy as np

from test_utils import Tester, TestingParameters

import time
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
import os

NUM_PARALLEL_JOBS = 10
PARALLEL_TRAIN = False

device = "cpu"


def add_parser_cmds(arg_parser):
    # Getting params
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "lpopl_ppo", "lpopl_dsac", "zero_shot_transfer", "random_transfer"]
    rl_algos = ["dqn", "dsac"]
    train_types = [
        "sequence",
        'test_until',
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
    parser.add_argument('--resume', default=False, action="store_true",
                        help='Whether to resume from a checkpoint or not.')
    parser.add_argument('--game_name', default="grid", type=str, choices=['grid', 'miniworld', 'miniworld_no_vis', 'miniworld_simp_no_vis'],
                        help='Name of the game.')
    parser.add_argument('--run_subfolder', default=None, required=False, type=str,
                        help='Name of the run. Used to save the results in a separate sub folder.')

if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=zero_shot_transfer --train_type=no_orders --train_size=50 --test_type=soft --map=0 --prob=0.8 --run_id=0 --relabel_method=cluster --transfer_num_times=1

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    add_parser_cmds(parser)

    # add all fields of Learning Parameters to parser
    LEARNING_ARGS_PREFIX = "lp."
    add_fields_to_parser(parser, LearningParameters, prefix=LEARNING_ARGS_PREFIX)
    parser.add_argument("--ltl_id", default=0, type=int, required=True)

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

    train_envs, test_envs = generate_envs(parallel=PARALLEL_TRAIN, map_id=map_id, seed=args.run_id)

    # path for logger
    tb_log_path = os.path.join(
        args.save_dpath, "results", f"{args.game_name}_{args.domain_name}", f"{args.train_type}_p{args.prob}", 
        f"{args.algo}_{args.rl_algo}", f"map{map_id}", str(args.run_id), 
        f"alpha={'auto' if learning_params.auto_alpha else learning_params.alpha}",
    )
    if testing_params.custom_metric_folder is not None:
        tb_log_path = os.path.join(tb_log_path, testing_params.custom_metric_folder)
    
    # load the proper lp
    with open(os.path.join(tb_log_path, "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)
    
    # logger
    writer = SummaryWriter(log_dir=os.path.join(tb_log_path, "logs", f"policy_{args.ltl_id}"))
    logger = TensorboardLogger(writer)

    # dump lp again
    # with open(os.path.join(tb_log_path, "learning_params.pkl"), "wb") as f:
    #     pickle.dump(learning_params, f)

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
        logger=logger
    )
    tasks = tester.tasks

    # run training
    global_time_steps = 0
    with open(os.path.join(tb_log_path, "logs", f"policy{args.ltl_id}_status.txt"), "w") as f:
        f.write(f"{time.time()},started\n")
    train_buffer = VectorReplayBuffer(int(1e6), buffer_num=NUM_PARALLEL_JOBS if PARALLEL_TRAIN else 1)

    policy_id = args.ltl_id
    policy, ltl = load_individual_policy(
        tb_log_path, policy_id, 
        num_actions=test_envs.action_space[0].n, 
        num_features=test_envs.observation_space[0].shape[0], 
        learning_params=learning_params, 
        device=device)

    # logging
    writer.add_text("task", str(ltl))
    
    # reset with the correct ltl
    task_params = tester.get_task_params(ltl)
    train_envs.reset(options=dict(task_params=task_params))
    test_envs.reset(options=dict(task_params=task_params))
    print(f"Training Sub-Task {policy_id}:", ltl)
    print("Global Time Step:", global_time_steps)
    
    # training
    train_buffer.reset()
    train_collector = Collector(policy, train_envs, train_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # train_collector.collect(n_step=4800, random=True)
    trainer = OffpolicyTrainer(
        policy, 
        train_collector, 
        test_collector, 
        max_epoch=100, 
        step_per_epoch=5000,
        episode_per_test=20, 
        batch_size=64, 
        update_per_step=1,
        step_per_collect=12,
        logger=logger,
        test_in_train=False,
        stop_fn=lambda x: x >= 9.5, # mean test reward,
        save_best_fn=lambda x: save_individual_policy(tb_log_path, policy_id, ltl, policy),
        # save_checkpoint_fn=lambda epoch, env_step, grad_step: save_individual_policy(tb_log_path, policy_id, ltl, policy)
        show_progress=("SLURM_JOB_ID" not in os.environ)
    )

    train_result = trainer.run()
    global_time_steps += train_result["train_step"]

    print("Done! Final Training Result:")
    print(train_result)
    with open(os.path.join(tb_log_path, "logs", f"policy{args.ltl_id}_status.txt"), "a") as f:
        f.write(f"{time.time()},done\n")
