import json
import os

import gymnasium
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import load_individual_policy
from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

import gzip

# %%
from tianshou.data import Batch
import numpy as np

from test_utils import TaskLoader, TestingParameters

import time
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
import tqdm
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
    parser.add_argument('--verbose', '-v', action="store_true", help='Verbose printing of results.')
    parser.add_argument('--relabel_seed', type=int, default=42, help="Seed for relabeling.")

    args = parser.parse_args()
    # if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    # if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    # if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    # if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = 0
    map_id = args.map

    print("########################################")
    print("Running rollout for:")
    print("policy:", args.ltl_id)
    print("map:", map_id)
    print("seed:", args.relabel_seed)
    print("rollout method:", args.rollout_method)
    print("########################################")

    task_loader = TaskLoader(args)
    testing_params = TestingParameters(custom_metric_folder=args.run_subfolder)

    train_envs, test_envs = generate_envs(prob=args.prob,
        game_name=args.game_name, 
        parallel=PARALLEL_TRAIN, 
        map_id=map_id, 
        seed=args.relabel_seed,
        no_info=False
    )
    
    # load the proper lp
    with open(os.path.join(task_loader.get_save_path(), "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)
    print("Initialized Learning Params:", learning_params)
    
    # logger
    writer = task_loader.writer
    logger = task_loader.logger

    # tester
    tester = TaskLoader(args)
    tasks = tester.tasks

    # sampler
    env_size = test_envs.get_env_attr("size", 0)[0]
    state_space = gymnasium.spaces.Box(
        low=np.array([1, 1, 0]), 
        high=np.array([env_size - 1, env_size - 1, 359.9])
    )
    if args.rollout_method == "uniform":
        space_iter = BoxSpaceIterator(state_space, interval=[.5, .5, 45])
    elif args.rollout_method == "random":
        space_iter = RandomIterator(state_space, num_samples=1000)
    else:
        raise NotImplementedError("Rollout method " + str(args.rollout_method) + " not implemented.")


    # run training
    ltl_id = args.ltl_id
    policy, ltl = load_individual_policy(
        task_loader.get_save_path(), ltl_id, 
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

    # clear output file
    outpath = os.path.join(
        task_loader.get_save_path(), 
        "classifier", 
        f"{args.rollout_method}_seed{args.relabel_seed}"
    )
    os.makedirs(outpath, exist_ok=True)
    outfile = os.path.join(outpath, f"policy{ltl_id}_rollout.json.gz")
    with gzip.open(outfile, 'wt', encoding='UTF-8') as f:
        json.dump({}, f)
    
    if args.render:
        test_envs.render()
        input("Press Enter When Ready...")

    # tqdm pretty print bar
    print("Expected total # of samples:", space_iter.__len__())
    if os.environ.get("SLURM_ARRAY_TASK_ID") is None:
        space_iter = tqdm.tqdm(space_iter)
    
    # begin rollout
    begin_time = time.time()
    result = {}
    for loc in space_iter:
        if loc is not None:
            x, y, angle = loc
            task_params.init_loc = [x, y]
            task_params.init_angle = angle
        obs, info = test_envs.reset(options=dict(task_params=task_params))

        dfa = DFA(ltl)
        init_loc = ", ".join([f"{item:.3f}" for item in info[0]['loc']])
        original_state = dfa.state
        if args.render:
            test_envs.render()
        for i in range(1500):
            a = policy.forward(Batch(obs=obs, info=info)).act
            obs, reward, term, trunc, info = test_envs.step(a.numpy())
            if args.render:
                test_envs.render()
                state = test_envs.get_env_attr("curr_state", 0)[0]
                print(state)
                input("Press Enter When Ready...")
            if term or trunc:
                true_prop = info[0]['true_props']
                success = info[0]['dfa_state'] != -1 and not trunc
                state = test_envs.get_env_attr("curr_state", 0)[0]
                result[init_loc] = {
                    "success": success,
                    "true_proposition": info[0]['true_props'] if success else '',
                    "steps": i + 1, 
                    "final_state": state,
                    "self_edge": info[0]['self_edge'],
                    "edge": info[0]['traversed_edge']
                }
                if args.verbose or args.render:
                    print(f"{x:.2f}, {y:.2f}, {angle}", result[init_loc])
                break

    end_time = time.time()
    os.makedirs(os.path.join(task_loader.get_save_path(), "classifier"), exist_ok=True)
    report_json = {
        "ltl": ltl,
        "time_spent": end_time - begin_time,
        "policy_last_updated": os.path.getmtime(os.path.join(task_loader.get_save_path(), "policies", f"{ltl_id}_ckpt.pth")),
        "rollout_method": args.rollout_method,
        "results": result
    }
    with gzip.open(outfile, 'wt', encoding='UTF-8') as f:
        json.dump(report_json, f)
    print("Saved rollout result to", outfile)
