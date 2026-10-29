#!/bin/bash
#SBATCH --job-name=dinov3-pretrain
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=196GB
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

# Pretrain RetinaNet with a DINOv3 backbone (dinov3-pretrain.yaml) and export
# the best checkpoint to <log_dir>/hf_model. Arguments are passed to
# `deepforest train`, e.g. data paths as Hydra overrides:
#   sbatch dinov3_pretrain_b200.sbatch train.csv_file=... train.root_dir=...

printenv | grep -i slurm | sort

source .venv/bin/activate

# DDP hangs at NCCL init on the B200 nodes unless NVLS is disabled
export NCCL_NVLS_ENABLE=0
export COMET_PROJECT="${COMET_PROJECT:-dinov3-lidar}"
srun --kill-on-bad-exit=1 uv run deepforest --config-name=dinov3-pretrain train --comet --export-hf "$@"
