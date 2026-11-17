#!/usr/bin/env bash
#SBATCH --job-name=milliontrees-eval
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=196GB
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

# Evaluation takes about 1 minute on NEON and under 20 minutes on MillionTrees

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"
source .venv/bin/activate

srun --kill-on-bad-exit=1 uv run deepforest --config-name "${MODEL_CONFIG:?MODEL_CONFIG not set}" \
  evaluate "$@"
