#!/bin/bash

datasets=('adult' 'mimic' 'fico' 'german' 'compas' 'diabetes')
methods=('lime' 'dice' 'kernelshap' 'treeshap' 'sev')

# Submit each job to Slurm
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        PARAMS="--dataset $dataset --method $method"
        sbatch \
        --job-name="exp7" \
        --output="../Results/Exp7_new/out/test_%j.out" \
        --error="../Results/Exp7_new/err/test_%j.err" \
        --wrap="python3 ../Code/Experiment7.py $PARAMS"
    done
done