#!/bin/bash

datasets=('adult' 'compas' 'german' 'mimic' 'diabetes' 'fico')
models=('lr' 'mlp' 'gbdt')
Opts=('alloptplus' 'alloptminus' 'volopt')

# Submit each job to Slurm
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for Opt in "${Opts[@]}"; do
        if [[ $model != "lr" && $Opt == "volopt" ]]; then
            continue
        fi
        PARAMS="--dataset $dataset --model $model --Optimized_method $Opt"
        
        sbatch \
        --job-name="experiment_1" \
        --output="../Results/Exp1/out/test_%j.out" \
        --error="../Results/Exp1/err/test_%j.err" \
        --wrap="python3 ../Code/Experiment1.py $PARAMS"
    done
  done
done
