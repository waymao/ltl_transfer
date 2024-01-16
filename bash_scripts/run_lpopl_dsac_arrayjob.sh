#!/bin/bash
#SBATCH -n 36
#SBATCH -N 1
#SBATCH --mem=36G
####SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-8


# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-dsac-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-dsac-%A-%a.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 1` # first index
j=`expr $SLURM_ARRAY_TASK_ID / 1`
k=`expr $j % 3` # second index
l=`expr $j / 3`
m=`expr $l % 3` # third index

train_types=( "sequence" )  # "hard" "mixed" "soft_strict" "soft" "no_orders"
train_type=${train_types[$i]}

# map=16
maps=( 1 13 15 )  # 0 1 5 6
map=${maps[$k]}

prob=1.0
#probs=( 1.0 )  # 0.9 0.8 0.7 0.6 0.5
#prob=${probs[$m]}

alphas=( 0.05 0.03 0.01 )
alpha=${alphas[$m]}
#alpha=0.1

#seeds=( 0 1 2 42 111 )
#seed=${seeds[$m]}
seed=42

algo="lpopl"
rl_algo="dsac"
game_name="miniworld"
train_size=50
# total_steps=2000000
# incremental_steps=2500000
save_dpath="$HOME/data/shared/ltl-transfer-pytorch/3"

source /users/ywei75/.bashrc
conda activate ltl

#PYGLET_HEADLESS="true" PYGLET_HEADLESS_DEVICE=0 python3 run_experiments.py --algo=$algo --rl_algo=$rl_algo --train_type=$train_type --train_size=$train_size --map=$map --prob=$prob --total_steps=8000000 --run_id=$seed --game_name $game_name --device cuda --alpha=$alpha
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    src/run_experiments.py \
    --algo=$algo --rl_algo=dsac --train_type=$train_type \
    --train_size=$train_size --map=$map --prob=$prob \
    --total_steps=$total_steps --incremental_steps=$incremental_steps \
    --save_dpath=$save_dpath --alpha=$alpha --run_id=$seed \
    --device=cpu --game_name="cluster1"
