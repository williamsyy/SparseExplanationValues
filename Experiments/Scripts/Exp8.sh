#!/bin/bash

datasets=('adult' 'compas' 'mimic' 'german' 'diabetes' 'fico')

# Submit each job to Slurm
for dataset in "${datasets[@]}"; do
    PARAMS="--data_name $dataset"
    
    sbatch \
    --job-name="exp9_$dataset" \
    -n 8 \
    --output="../Results/Exp9/out/test_%j.out" \
    --error="../Results/Exp9/err/test_%j.err" \
    --wrap="python3 ../Code/Experiment9.py $PARAMS"
done