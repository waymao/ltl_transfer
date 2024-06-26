#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH -t 24:00:00
####### 460 tasks per map.
#SBATCH --array=0-1840
##SBATCH --array=0-27

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-train-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-train-%A-%a.out

#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu

map=13
train_size=50
train_type="mixed"

# domain name
prob=1.0
domain_name="spot"
game_name="miniworld_simp_no_vis"
alpha=0.03

# run id array
# run_id=`expr $SLURM_ARRAY_TASK_ID / 460`
# map=21

# map array
map_ids=( 21 22 23 )
run_id=0
rl_algo="dsac"

# count the number of policies in the save path
save_path=$HOME/data/shared/ltl-transfer-ts/saves/${game_name}_${domain_name}_p${prob}/${train_type}_50/$rl_algo/map${map_ids[0]}/${run_id}/policies/
num_policies="$(ls $save_path | wc -l)"

echo "Detected $num_policies policies."

######### PREDEFINED ARRAYS #########
# ltl id
ltl_id=`expr $SLURM_ARRAY_TASK_ID % $num_policies`
i=`expr $SLURM_ARRAY_TASK_ID / $num_policies`

# map
map_id=$i
map=${map_ids[$map_id]}

echo $map


source $HOME/.bashrc
conda activate ltl

#echo PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
#        --train_size $train_size --rl_algo dsac --map $map --ltl_id $ltl_id \
#        --game_name miniworld_simp_no_vis --train_type $train_type \
#        --save_dpath=$HOME/data/shared/ltl-transfer-ts

PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
        --train_size $train_size --rl_algo dsac --domain_name $domain_name \
        --map $map --ltl_id $ltl_id \
        --game_name $game_name --train_type $train_type \
        --lp.alpha=$alpha --prob=$prob --run_id $run_id \
        --save_dpath=$HOME/data/shared/ltl-transfer-ts
