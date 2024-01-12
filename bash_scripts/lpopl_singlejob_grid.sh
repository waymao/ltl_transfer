#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=10G
#SBATCH -t 99:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/job-%j.err
#SBATCH -o sbatch_out/job-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=xinyu_liu@brown.edu

export PYTHONUNBUFFERED=TRUE

algo="lpopl"
rl_algo="dsac"
train_type="sequence"
train_size=50
map=0
prob=1.0
total_steps=800000
incremental_steps=50000
save_dpath="$HOME/data/shared/ltl-transfer-pytorch/3"
device="cuda"

module load miniconda/4.12.0
source /oscar/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate ltl_pt

xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python src/run_experiments.py --algo=$algo --rl_algo=$rl_algo --train_type=$train_type --train_size=$train_size --map=$map --prob=$prob --total_steps=$total_steps --incremental_steps=$incremental_steps --save_dpath=$save_dpath --device=$device
