#!/bin/bash

datasets=('adult' 'compas' 'mimic' 'german')

# Submit each job to Slurm
for dataset in "${datasets[@]}"; do
    PARAMS="--data_name $dataset"
    
    sbatch \
    --job-name="exp6" \
    --output="../Results/Exp6/out/test_%j.out" \
    --error="../Results/Exp6/err/test_%j.err" \
    --wrap="python3 ../Code/Experiment6.py $PARAMS"
done