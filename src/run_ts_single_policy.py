
# %%
import os
from torch_policies.learning_params import LearningParameters, \
    add_fields_to_parser, get_learning_parameters

from ts_utils.ts_policy_bank import load_individual_policy, save_individual_policy
from ts_utils.ts_envs import generate_envs
from ts_utils.ts_argparse import add_parser_cmds

# %%
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer

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

    train_envs, test_envs = generate_envs(prob=args.prob,game_name=args.game_name, parallel=PARALLEL_TRAIN, map_id=map_id, seed=args.run_id)

    
    # loading tasks
    task_loader = TaskLoader(args, create_logger=True)
    tasks = task_loader.tasks
    logger = task_loader.logger
    writer = task_loader.writer

    # load the proper lp
    with open(os.path.join(task_loader.get_save_path(), "learning_params.pkl"), "rb") as f:
        learning_params = pickle.load(f)

    # run training
    global_time_steps = 0
    with open(os.path.join(task_loader.get_save_path(), "logs", f"policy{args.ltl_id}_status.txt"), "w") as f:
        f.write(f"{time.time()},started\n")
    train_buffer = VectorReplayBuffer(int(1e6), buffer_num=NUM_PARALLEL_JOBS if PARALLEL_TRAIN else 1)

    policy_id = args.ltl_id
    policy, ltl = load_individual_policy(
        task_loader.get_save_path(), policy_id, 
        num_actions=test_envs.action_space[0].n, 
        num_features=test_envs.observation_space[0].shape[0], 
        learning_params=learning_params, 
        device=device)

    # logging
    writer.add_text("task", str(ltl))
    
    # reset with the correct ltl
    task_params = task_loader.get_task_params(ltl)
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
        stop_fn=lambda rew: rew >= 9.5, # mean test reward,
        save_best_fn=lambda x: save_individual_policy(task_loader.get_save_path(), policy_id, ltl, policy),
        # save_checkpoint_fn=lambda epoch, env_step, grad_step: save_individual_policy(task_loader.get_save_path(), policy_id, ltl, policy)
        show_progress=("SLURM_JOB_ID" not in os.environ)
    )

    train_result = trainer.run()
    global_time_steps += train_result["train_step"]

    print("Done! Final Training Result:")
    print(train_result)
    with open(os.path.join(task_loader.get_save_path(), "logs", f"policy{args.ltl_id}_status.txt"), "a") as f:
        f.write(f"{time.time()},done\n")
