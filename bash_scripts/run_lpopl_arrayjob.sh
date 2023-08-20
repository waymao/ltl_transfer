#!/bin/bash
#SBATCH -n 16
#SBATCH --mem=99G
#SBATCH -t 99:00:00
#SBATCH --array=0-19

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-%A-%a.err
#SBATCH -o sbatch_out/arrayjob-%A-%a.out

export PYTHONUNBUFFERED=TRUE

# Convert 1D indexing to 2D
i=`expr $SLURM_ARRAY_TASK_ID % 5`
j=`expr $SLURM_ARRAY_TASK_ID / 5`
k=`expr $j % 4`
#l=`expr $j / 3`
#m=`expr $l % 1`

train_types=( "hard" "mixed" "soft_strict" "soft" "no_orders" )
train_type=${train_types[$i]}

maps=( 0 1 5 6 )
map=${maps[$k]}

algo="lpopl"
train_size=50
total_steps=800000
save_dpath="$HOME/data/shared/ltl-transfer"

module load anaconda/2022.05
source /oscar/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate lpopl

python src/run_experiments.py --algo=$algo --train_type=$train_type --train_size=$train_size --map=$map --total_steps=$total_steps --save_dpath=$save_dpath
