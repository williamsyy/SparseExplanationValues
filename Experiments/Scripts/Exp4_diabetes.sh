params=(
    "--dataset diabetes --model lr --Optimized_method alloptplus --sev_penalty 0.01 --positive_penalty 10"
    "--dataset diabetes --model lr --Optimized_method alloptminus --sev_penalty 0.01 --positive_penalty 10"
    "--dataset diabetes --model lr --Optimized_method volopt --sev_penalty 0.1 --positive_penalty 0.1"
    "--dataset diabetes --model mlp --Optimized_method alloptplus --sev_penalty 0.01 --positive_penalty 1"
    "--dataset diabetes --model mlp --Optimized_method alloptminus --sev_penalty 0.01 --positive_penalty 10"
    "--dataset diabetes --model gbdt --Optimized_method alloptplus --sev_penalty 0.01 --positive_penalty 0.1"
    "--dataset diabetes --model gbdt --Optimized_method alloptminus --sev_penalty 0.01 --positive_penalty 1"
)

for param in "${params[@]}"; do
    sbatch \
    --job-name="experiment_4" \
    --output="../Results/Exp4/out/test_%j.out" \
    --error="../Results/Exp4/err/test_%j.err" \
    --wrap="python3 ../Code/Experiment4.py $param"
done