import json
import os

import gymnasium
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import create_discrete_sac_policy, TianshouPolicyBank, load_ts_policy_bank, load_individual_policy, save_individual_policy
from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

# %%
from tianshou.policy import PPOPolicy, DiscreteSACPolicy, TD3Policy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch
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

from rollout_utils.sampler import BoxSpaceIterator, RandomIterator

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
    parser.add_argument('--run_prefix', type=str,
                        help='Location of the bank and the learning parameters.')
    parser.add_argument('--ltl_id', type=int, required=True, help='Policy ID to demo')
    parser.add_argument('--no_deterministic_eval', action="store_true", help='Whether to run deterministic evaluation or not.')
    parser.add_argument('--rollout_method', type=str, default="uniform", choices=['uniform', 'random'], help='How to rollout the policy.')
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
        no_info=False
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

    # sampler
    env_size = test_envs.get_env_attr("size", 0)[0]
    state_space = gymnasium.spaces.Box(
        low=np.array([1, 1, -180]), 
        high=np.array([env_size - 1, env_size - 1, 180])
    )
    if args.rollout_method == "uniform":
        space_iter = BoxSpaceIterator(state_space, interval=[.5, .5, 30])
    elif args.rollout_method == "random":
        space_iter = RandomIterator(state_space, num_samples=100)
    else:
        raise NotImplementedError("Rollout method " + str(args.rollout_method) + " not implemented.")


    # run training
    global_time_steps = 0
    with open(os.path.join(tb_log_path, "logs", f"policy{args.ltl_id}_status.txt"), "w") as f:
        f.write(f"{time.time()},started\n")
    train_buffer = VectorReplayBuffer(int(1e6), buffer_num=NUM_PARALLEL_JOBS if PARALLEL_TRAIN else 1)

    ltl_id = args.ltl_id
    policy, ltl = load_individual_policy(
        tb_log_path, ltl_id, 
        num_actions=test_envs.action_space[0].n, 
        num_features=test_envs.observation_space[0].shape[0], 
        learning_params=learning_params, 
        device=device)

    print("Running policy", ltl)
    
    # reset with the correct ltl
    task_params = tester.get_task_params(ltl)
    test_envs.reset()
    test_envs.get_env_attr("unwrapped", 0)
    
    # collecting results
    # set policy to be deterministic if set up to do so
    if not args.no_deterministic_eval:
        policy.training = False
        policy.eval()

    # collect and rollout
    #uncomment for the coordinates
    results = {}
    if args.render:
        test_envs.render()
        input("Press Enter When Ready...")
    for x, y, angle in space_iter:
        task_params.init_loc = [x, y]
        obs, info = test_envs.reset(options=dict(task_params=task_params))
        if args.render:
            test_envs.render()
        for i in range(1500):
            a = policy.forward(Batch(obs=obs, info=info)).act
            obs, reward, term, trunc, info = test_envs.step(a.numpy())
            if args.render:
                test_envs.render()
            if term or trunc:
                true_prop = info[0]['true_props']
                success = info[0]['dfa_state'] != -1 and not trunc
                results[f"{x}, {y}"] = {
                    "success": success,
                    "true_proposition": info[0]['true_props'] if success else '',
                    "steps": i + 1, 
                    # "new_ltl_goal": info[0]['ltl_goal'],
                }
                print(f"{x:.2f}, {y:.2f}", results[f"{x}, {y}"])
                break

    print(results)
    os.makedirs(os.path.join(tb_log_path, "classifier"), exist_ok=True)
    with open(os.path.join(tb_log_path, "classifier", f"policy{args.ltl_id}_status.json"), "w") as f:
        json.dump(results, f)
