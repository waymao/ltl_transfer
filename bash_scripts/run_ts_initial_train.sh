#!/bin/bash
#SBATCH -n 52
#SBATCH -N 1
#SBATCH --mem=16G

#SBATCH -t 48:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-dsac-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-dsac-%A-%a.out

#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu


map=13
run_id=1
train_size=50
train_type="sequence"

i=$SLURM_ARRAY_TASK_ID

source /users/ywei75/.bashrc
conda activate ltl

PYGLET_HEADLESS=true python3 run_ts_state_policybank.py \
        --train_size $train_size --rl_algo dsac --map $map \
        --game_name miniworld_simp_no_vis --train_type $train_type \
        --save_dpath=/users/ywei75/data/shared/ltl-transfer-ts

