#!/bin/bash

# datasets=('adult' 'compas' 'german' 'mimic' 'diabetes' 'fico')
datasets=('german')
models=('l2lr' 'l1lr' 'mlp' 'gbdt')
SEV_modes=('plus' 'minus')

# Submit each job to Slurm
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for SEV_mode in "${SEV_modes[@]}"; do
        PARAMS="--dataset $dataset --model $model --SEV_mode $SEV_mode --repeat 10"
        
        sbatch \
        --job-name="experiment_0" \
        --output="../Results/Exp0/out/test_%j.out" \
        --error="../Results/Exp0/err/test_%j.err" \
        --wrap="python3 ../Code/Experiment0.py $PARAMS"
    done
  done
done
