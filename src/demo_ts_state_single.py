import os
from ltl.dfa import DFA
from utils.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import load_individual_policy
from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

# %%
from tianshou.data import VectorReplayBuffer, Batch

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
    parser.add_argument('--run_prefix', type=str,
                        help='Location of the bank and the learning parameters.')
    parser.add_argument('--ltl_id', type=int, required=True, help='Policy ID to demo')
    parser.add_argument('--no_deterministic_eval', action="store_true", help='Whether to run deterministic evaluation or not.')
    parser.add_argument('--init_state', type=str, help='Initial state of the agent', required=False, default=None)

    args = parser.parse_args()
    # if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    # if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    # if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    # if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    map_id = args.map
    
    # enable rendering
    if args.game_name == "miniworld_no_vis":
        save_game_name = "miniworld_simp_no_vis"

    # learning params
    learning_params = get_learning_parameters(
        policy_name=args.rl_algo, 
        game_name=save_game_name,
        **{
            key.removeprefix(LEARNING_ARGS_PREFIX): val 
            for (key, val) in args._get_kwargs() 
            if key.startswith(LEARNING_ARGS_PREFIX)
        }
    )
    testing_params = TestingParameters(custom_metric_folder=args.run_subfolder)
    print("Initialized Learning Params:", learning_params)

    train_envs, test_envs = generate_envs(
        prob=args.prob,game_name=args.game_name, 
        parallel=PARALLEL_TRAIN, map_id=map_id, seed=args.run_id,
        no_info=False
    )

    # tester
    task_loader = TaskLoader(args)
    tasks = task_loader.tasks
    
    # load the proper lp
    with open(os.path.join(task_loader.get_save_path(), "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)


    # run training
    global_time_steps = 0
    with open(os.path.join(task_loader.get_save_path(), "logs", f"policy{args.ltl_id}_status.txt"), "w") as f:
        f.write(f"{time.time()},started\n")
    train_buffer = VectorReplayBuffer(int(1e6), buffer_num=NUM_PARALLEL_JOBS if PARALLEL_TRAIN else 1)

    ltl_id = args.ltl_id
    policy, ltl = load_individual_policy(
        task_loader.get_save_path(), ltl_id, 
        num_actions=test_envs.action_space[0].n, 
        num_features=test_envs.observation_space[0].shape[0], 
        learning_params=learning_params, 
        device=device)

    print("Running policy", ltl)
    
    # reset with the correct ltl
    task_params = task_loader.get_task_params(ltl)
    test_envs.reset()
    
    # collecting results
    # set policy to be deterministic if set up to do so
    if not args.no_deterministic_eval:
        policy.training = False
        policy.eval()

    # collect and rollout
    #uncomment for the coordinates
    # for x in range(2, 10, 0.1):
    if args.init_state is not None: 
        if "miniworld" in args.game_name:
            task_params.init_loc = map(float, args.init_state.split(",")[:2])
            task_params.init_angle = float(args.init_state.split(",")[2])
    for x in range(1000):
        obs, info = test_envs.reset(options=dict(task_params=task_params))
        print(args.init_state, info[0]['loc'])
        test_envs.render()
        for i in range(200):
            a = policy.forward(Batch(obs=obs, info=info)).act
            obs, reward, term, trunc, info = test_envs.step(a.numpy())
            test_envs.render()
            if term or trunc:
                print("Finished. info:", info)
                break
