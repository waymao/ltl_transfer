
# %%
import os
from ltl.dfa import DFA
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import create_discrete_sac_policy, TianshouPolicyBank, load_ts_policy_bank
from ts_utils.ts_argparse import add_parser_cmds

# %%
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer

from test_utils import Tester, TestingParameters

import time
import argparse
import pickle

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
import os

from ts_utils.ts_envs import generate_envs

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

    train_envs, test_envs = generate_envs(prob=args.prob,game_name=args.game_name, parallel=PARALLEL_TRAIN, map_id=map_id, seed=args.run_id)

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
    writer = SummaryWriter(log_dir=tb_log_path)
    logger = TensorboardLogger(writer)

    # dump lp again
    with open(os.path.join(tb_log_path, "learning_params.pkl"), "wb") as f:
        pickle.dump(learning_params, f)

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

    # initalize policy bank
    if not args.resume:
        policy_bank = TianshouPolicyBank()

        for task in tasks:
            dfa = DFA(task)
            for ltl in dfa.ltl2state.keys():
                policy = create_discrete_sac_policy(
                    num_actions=test_envs.action_space[0].n, 
                    num_features=test_envs.observation_space[0].shape[0], 
                    hidden_layers=[256, 256, 256],
                    learning_params=learning_params,
                    device=device
                )
                policy_bank.add_LTL_policy(ltl, policy)
    else:
        policy_bank = load_ts_policy_bank(
            tb_log_path, 
            num_actions=test_envs.action_space[0].n,
            num_features=test_envs.observation_space[0].shape[0],
            hidden_layers=[256, 256, 256],
            learning_params=learning_params,
            load_classifier=None,
            device=device
        )

    # run training
    global_time_steps = 0
    with open(os.path.join(tb_log_path, "policy_log.txt"), "w") as f:
        f.write("ltl,global_time_steps,time\n")
    total_tasks = len(policy_bank.get_all_policies())
    train_buffer = VectorReplayBuffer(int(1e6), buffer_num=NUM_PARALLEL_JOBS if PARALLEL_TRAIN else 1)
    for ltl, i in policy_bank.policy2id.items():
        policy = policy_bank.policies[i]
        # logging
        with open(os.path.join(tb_log_path, "policy_log.txt"), "a") as f:
            f.write(f"\"{ltl}\",{global_time_steps},{time.time()}\n")
        writer.add_text("task", str(ltl))

        # skip if it's a dummy policy
        if ltl == "True" or ltl == "False": continue
        
        # reset with the correct ltl
        task_params = tester.get_task_params(ltl)
        train_envs.reset(options=dict(task_params=task_params))
        test_envs.reset(options=dict(task_params=task_params))
        print(f"Training Sub-Task {i + 1}/{total_tasks}:", ltl)
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
            save_best_fn=lambda x: policy_bank.save_individual_policy(tb_log_path, i),
            save_checkpoint_fn=lambda epoch, env_step, grad_step: policy_bank.save_ckpt(tb_log_path)
        )

        train_result = trainer.run()
        global_time_steps += train_result["train_step"]

