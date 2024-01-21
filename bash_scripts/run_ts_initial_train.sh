#!/bin/bash
#SBATCH -n 36
#SBATCH -N 1
#SBATCH --mem=6G

#SBATCH -t 48:00:00

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-dsac-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-dsac-%A-%a.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu


python3 run_ts_state_policybank.py --train_size 50 --rl_algo dsac --map 13 --game_name miniworld_simp_no_vis --train_type mixed
