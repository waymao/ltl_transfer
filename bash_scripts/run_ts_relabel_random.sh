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

train_size=50
train_type="mixed"

rollout_method="random"

# domain name
prob=1.0
domain_name="spot"
game_name="miniworld_simp_no_vis"
alpha=0.03

map=21
run_id=0

######### PREDEFINED ARRAYS #########
# ltl id
ltl_id=`expr $SLURM_ARRAY_TASK_ID % 460`
i=`expr $SLURM_ARRAY_TASK_ID / 460`


# map
map_ids=( 21 22 23 )

map_len=${#map_ids[@]}
map_id=`expr $i % $map_len`
map=${map_ids[$map_id]}

j=`expr $i / $map_len`

echo i=$i
echo map_len=$map_len
echo map=$map


# relabel method
seeds=( 0 1 2 )

seed_len=${#seeds[@]}
seed_id=`expr $j % $seed_len`
seed=${seeds[$seed_id]}

echo j=$j
echo seed_len=$seed_len
echo seed=$seed

######### END PREDEFINED ARRAYS #########


source $HOME/.bashrc
conda activate ltl


# PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
#         --train_size $train_size --rl_algo dsac --map $map --ltl_id $ltl_id \
#         --game_name miniworld_simp_no_vis --train_type $train_type \
#         --save_dpath=$HOME/data/shared/ltl-transfer-ts

PYGLET_HEADLESS=true python3 run_ts_single_rollout.py \
        --save_dpath=$HOME/data/shared/ltl-transfer-ts \
        --game_name miniworld_simp_no_vis \
        --domain_name $domain_name --prob $prob \
        --map $map --train_type $train_type \
        --run_id $run_id --relabel_seed $seed_id \
        --ltl $ltl_id --rollout_method $rollout_method
