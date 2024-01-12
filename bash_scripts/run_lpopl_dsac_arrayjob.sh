#!/bin/bash
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --mem=36G
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --array=0-9


# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-dsac-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-dsac-%A-%a.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yichen_wei@brown.edu

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 1`
j=`expr $SLURM_ARRAY_TASK_ID / 1`
k=`expr $j % 10`
l=`expr $j / 10`
m=`expr $l % 1`

train_types=( "sequence" )  # "hard" "mixed" "soft_strict" "soft" "no_orders"
train_type=${train_types[$i]}

map=16
#maps=( 0 1 5 6 )  # 0 1 5 6
#map=${maps[$k]}

prob=1.0
#probs=( 1.0 )  # 0.9 0.8 0.7 0.6 0.5
#prob=${probs[$m]}

alphas=( 100 30 10 3 1 0.3 0.1 0.05 0.03 0.01)
alpha=${alphas[$m]}
#alpha=0.1

#seeds=( 0 1 2 42 111 )
#seed=${seeds[$m]}
seed=42

algo="lpopl"
rl_algo="dsac"
game_name="miniworld"
train_size=50
total_steps=2000000
incremental_steps=2500000
save_dpath="$HOME/data/shared/ltl-transfer-pytorch/3"

module load miniconda/4.12.0
source /oscar/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate ltl_pt

PYGLET_HEADLESS="true" PYGLET_HEADLESS_DEVICE=0 python3 run_experiments.py --algo=$algo --rl_algo=$rl_algo --train_type=$train_type --train_size=$train_size --map=$map --prob=$prob --total_steps=8000000 --run_id=$seed --game_name $game_name --device cuda --alpha=$alpha
#python src/run_experiments.py --algo=$algo --rl_algo=dsac --train_type=$train_type --train_size=$train_size --map=$map --prob=$prob --total_steps=$total_steps --incremental_steps=$incremental_steps --save_dpath=$save_dpath --alpha=$alpha --run_id=$seed
