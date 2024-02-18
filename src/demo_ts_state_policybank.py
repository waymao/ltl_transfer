
# %%
import os
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, add_fields_to_parser

from ts_utils.ts_policy_bank import TianshouPolicyBank, load_ts_policy_bank
from ts_utils.ts_argparse import add_parser_cmds
# %%
from ts_utils.ts_envs import generate_envs

from test_utils import TaskLoader, TestingParameters

import argparse
import pickle

from tianshou.data import Batch
import os
from tianshou.policy import BasePolicy

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

    parser.add_argument('--ltl_id', type=int, required=True, help='Policy ID to demo')
    parser.add_argument('--no_deterministic_eval', action="store_true", help='Whether to run deterministic evaluation or not.')
    args = parser.parse_args()
    # if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    # if args.train_type not in train_types: raise NotImplementedError("Training tasks " + str(args.train_type) + " hasn't been defined yet")
    # if args.test_type not in test_types: raise NotImplementedError("Test tasks " + str(args.test_type) + " hasn't been defined yet")
    # if not(-2 <= args.map < 20): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = 0
    map_id = args.map

    save_path = args.run_prefix

    # learning params
    with open(os.path.join(save_path, "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)
    testing_params = TestingParameters(custom_metric_folder=args.run_subfolder)
    print("Initialized Learning Params:", learning_params)

    train_envs, test_envs = generate_envs(prob=args.prob,
        game_name=args.game_name, 
        parallel=False, 
        map_id=map_id, 
        seed=args.run_id,
    )

    # tester
    tester = TaskLoader(
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
        logger=None
    )
    tasks = tester.tasks

    # initalize policy bank
    policy_bank: TianshouPolicyBank = load_ts_policy_bank(
        os.path.join(save_path),
        num_actions=test_envs.action_space[0].n,
        num_features=test_envs.observation_space[0].shape[0],
    )
    # sanity check to make sure everything is in the policy bank.
    for task in tasks:
        dfa = DFA(task)
        for ltl in dfa.ltl2state.keys():
            if ltl != 'True' and ltl != 'False': 
                assert ltl in policy_bank.policy2id, \
                    ("LTL " + str(ltl) + " not found in policy bank.")
    assert args.ltl_id in policy_bank.policy2id.values(), "Policy ID not found in policy bank"

    # get the correct policy
    policy: BasePolicy = policy_bank.policies[args.ltl_id]
    ltl = policy_bank.policy_ltls[args.ltl_id]

    print("Running policy", ltl)
    
    # reset with the correct ltl
    task_params = tester.get_task_params(ltl)
    test_envs.reset()
    
    # collecting results
    # set policy to be deterministic if set up to do so
    if not args.no_deterministic_eval:
        policy.training = False
        policy.eval()

    # collect and rollout
    #uncomment for the coordinates
    # for x in range(2, 10, 0.1):
    for x in range(1000):
        # uncomment for a specific location
        # task_params.init_loc = [x, 5.0]
        obs, info = test_envs.reset(options=dict(task_params=task_params))
        test_envs.render()
        for i in range(200):
            print(obs)
            a = policy.forward(Batch(obs=obs, info=info)).act
            obs, reward, term, trunc, info = test_envs.step(a.numpy())
            test_envs.render()
            if term or trunc:
                break
