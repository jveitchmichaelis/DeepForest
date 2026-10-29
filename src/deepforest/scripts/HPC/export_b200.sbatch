#!/bin/bash
#SBATCH --job-name=deepforest-export
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

# Export a Lightning checkpoint as a Hugging Face model. Arguments are
# passed to `deepforest export`, e.g.:
#   sbatch export_b200.sbatch logs/run/checkpoints/last.ckpt -o logs/run/hf_model_last

printenv | grep -i slurm | sort

source .venv/bin/activate

srun --kill-on-bad-exit=1 uv run deepforest export "$@"
