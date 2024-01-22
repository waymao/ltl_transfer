#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
##SBATCH --array=0-648
#SBATCH --array=0-27

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ~/scratch/ltl_ts/arrayjob-train-%A-%a.err
#SBATCH -o ~/scratch/ltl_ts/arrayjob-train-%A-%a.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu

map=13
run_id=1
train_size=50
train_type="sequence"

ltl_id=`expr $SLURM_ARRAY_TASK_ID`

source /users/ywei75/.bashrc
conda activate ltl

#echo PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
#        --train_size $train_size --rl_algo dsac --map $map --ltl_id $ltl_id \
#        --game_name miniworld_simp_no_vis --train_type $train_type \
#        --save_dpath=/users/ywei75/data/shared/ltl-transfer-ts

PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
        --train_size $train_size --rl_algo dsac --map $map --ltl_id $ltl_id \
        --game_name miniworld_simp_no_vis --train_type $train_type \
        --save_dpath=/users/ywei75/data/shared/ltl-transfer-ts