#!/usr/bin/env bash
# Generates the three region datasets one after the other (each uses every core) and draws their figures. Rerunning it
# resumes a model from its checkpoint in scratch/. Logs go to scratch/sgra_<model>_dominated.log.
#
#     scripts/run_regions.sh [models...]        (default: disk funnel jet)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scratch
models=("$@"); [ ${#models[@]} -eq 0 ] && models=(disk funnel jet)
for model in "${models[@]}"; do
    julia -t auto --project=. scripts/sgra_regions.jl --model "$model" --resume 2>&1 | tee -a "scratch/sgra_${model}_dominated.log"
    julia -t auto --project=. scripts/plot_regions.jl "datasets/sgra_${model}_dominated" 2>&1 | tee -a "scratch/sgra_${model}_dominated.log"
done
