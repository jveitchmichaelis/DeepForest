#!/usr/bin/env bash
#SBATCH --job-name=milliontrees-train
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=512GB
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"
source .venv/bin/activate

export NCCL_NVLS_ENABLE=0
export COMET_PROJECT="${COMET_PROJECT:-milliontrees}"

srun --kill-on-bad-exit=1 uv run deepforest --config-name "${MODEL_CONFIG:?MODEL_CONFIG not set}" \
  train --comet --export-hf "$@"
