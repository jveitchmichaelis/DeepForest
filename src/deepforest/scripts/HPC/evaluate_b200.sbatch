#!/bin/bash
#SBATCH --job-name=deepforest-evaluate
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

# Evaluate a model on a ground truth CSV. Arguments are passed to
# `deepforest evaluate`, e.g.:
#   sbatch evaluate_b200.sbatch test.csv --root-dir images -o results.csv model.name=...

printenv | grep -i slurm | sort

source .venv/bin/activate

if [[ -f "$HOME/.secrets/hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token")"
fi

srun --kill-on-bad-exit=1 uv run deepforest evaluate "$@"
