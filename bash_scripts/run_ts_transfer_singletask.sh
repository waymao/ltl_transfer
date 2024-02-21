#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -t 48:00:00
##SBATCH --array=0-1199
#SBATCH --array=0-27

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-train-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-train-%A-%a.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu

train_size=50
train_type="mixed"

rollout_method="uniform"

# domain name
prob=1.0
domain_name="spot"
game_name="miniworld_simp_no_vis"
alpha=0.03

map_ids=( 21 22 23 )
run_id=0
rl_algo="dsac"

num_tasks=100

######### PREDEFINED ARRAYS #########
# ltl id
task_id=`expr $SLURM_ARRAY_TASK_ID % $num_tasks`
i=`expr $SLURM_ARRAY_TASK_ID / $num_tasks`


# map
map_len=${#map_ids[@]}
map_id=`expr $i % $map_len`
map=${map_ids[$map_id]}

j=`expr $i / $map_len`

# echo i=$i
# echo map_len=$map_len
echo map=$map


# relabel method
relabel_args_list=( 
        "--relabel_method knn_random --relabel_seed 0" 
        "--relabel_method knn_random --relabel_seed 1" 
        "--relabel_method knn_random --relabel_seed 2"
        "--relabel_method knn_uniform --relabel_seed 0" 
)
relabel_args_list_len=${#relabel_args_list[@]}
relabel_id=`expr $j % $relabel_args_list_len`
relabel_args=${relabel_args_list[$relabel_id]}

echo relabel_args=$relabel_args
# echo j=$j
############ END PREDEFINED ARRAYS #########



source $HOME/.bashrc
conda activate ltl


# PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
#         --train_size $train_size --rl_algo dsac --map $map --ltl_id $ltl_id \
#         --game_name miniworld_simp_no_vis --train_type $train_type \
#         --save_dpath=$HOME/data/shared/ltl-transfer-ts

cmd="PYGLET_HEADLESS=true python run_ts_transfer.py \
        --save_dpath=$HOME/data/shared/ltl-transfer-ts \
        --game_name miniworld_simp_no_vis \
        --domain_name $domain_name --prob $prob \
        --map $map --train_type $train_type \
        --run_id $run_id \
        $relabel_args \
        --task_id $task_id"
echo $cmd
echo
eval $cmd

