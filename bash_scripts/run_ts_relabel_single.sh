#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH -t 24:00:00
####### 460 tasks per map.
#SBATCH --array=0-1840
##SBATCH --array=0-27

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-relabel-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-relabel-%A-%a.out

#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu

map=13
train_size=50
train_type="mixed"

rollout_method="random"

# domain name
prob=1.0
domain_name="spot"
game_name="miniworld_simp_no_vis"
alpha=0.03

map=21
run_id=42

######### PREDEFINED ARRAYS #########
# run id array
# run_id=`expr $SLURM_ARRAY_TASK_ID / 460`

# map array
# map=`expr $SLURM_ARRAY_TASK_ID / 460 + 21`

# relabel method
k=`expr $SLURM_ARRAY_TASK_ID / 460`
relabel_methods=( "random" "uniform" )
relabel_method=${relabel_methods[$k]}

# ltl id
ltl_id=`expr $SLURM_ARRAY_TASK_ID % 460`
######### END PREDEFINED ARRAYS #########


source $HOME/.bashrc
conda activate ltl


# PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
#         --train_size $train_size --rl_algo dsac --map $map --ltl_id $ltl_id \
#         --game_name miniworld_simp_no_vis --train_type $train_type \
#         --save_dpath=$HOME/data/shared/ltl-transfer-ts

PYGLET_HEADLESS=true python run_ts_single_rollout.py \
        --save_dpath=$HOME/data/shared/ltl-transfer-ts \
        --game_name miniworld_simp_no_vis \
        --map $map --train_type $train_type \
        --run_id $run_id \
        --ltl $ltl_id --rollout_method $rollout_method
