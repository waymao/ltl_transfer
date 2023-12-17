#!/bin/bash

export PYGLET_HEADLESS=true 
export PYGLET_HEADLESS_DEVICE=0

for alpha in 10 3 1 0.3 0.1 0.05 0.03 0.01 0.005
do 
    echo "alpha: $alpha"
    python3 run_experiments.py \
        --algo=lpopl --rl_algo=dsac --train_type=sequence --train_size=50 \
        --prob=1.0 --total_steps=1000000 --run_id=42 --game_name miniworld \
        --device cuda --map=16 --alpha $alpha
done