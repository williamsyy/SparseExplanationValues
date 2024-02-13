#!/bin/bash

# datasets=('adult' 'compas' 'german' 'mimic' 'diabetes' 'fico')
datasets=('german')
models=('l2lr' 'l1lr' 'mlp' 'gbdt')
SEV_modes=('plus' 'minus')

# Submit each job to Slurm
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for SEV_mode in "${SEV_modes[@]}"; do
        PARAMS="--dataset $dataset --model $model --SEV_mode $SEV_mode"
        
        sbatch \
        --job-name="experiment_0_baseline" \
        --output="../Results/Exp0_Baseline/out/test_%j.out" \
        --error="../Results/Exp0_Baseline/err/test_%j.err" \
        --wrap="python3 ../Code/Experiment0-baseline.py $PARAMS"
    done
  done
done
