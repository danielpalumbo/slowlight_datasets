#!/usr/bin/env bash
# Traces the band-6 movies of the three region datasets one after the other (each uses every core), drawing each
# model's mp4 while the next one traces. Rerunning it resumes a model from its checkpoint in scratch/. Logs go to
# scratch/sgra_<model>_dominated_movie.log; the image cubes to movies/ (untracked).
#
#     scripts/run_movies.sh [models...]        (default: disk funnel jet)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scratch
models=("$@"); [ ${#models[@]} -eq 0 ] && models=(disk funnel jet)
for model in "${models[@]}"; do
    log="scratch/sgra_${model}_dominated_movie.log"
    julia -t auto --project=. scripts/movie_regions.jl --model "$model" --resume 2>&1 | tee -a "$log"
    julia --project=. scripts/draw_movie.jl "datasets/sgra_${model}_dominated" >> "$log" 2>&1 &
done
wait
