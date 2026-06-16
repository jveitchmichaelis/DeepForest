#!/bin/bash
#SBATCH --job-name=maskrcnn-oam
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err
#SBATCH --ntasks-per-node=1

printenv | grep -i slurm | sort

export COMET_PROJECT="deepforest-maskrcnn"

mkdir -p slurm_logs

srun uv run deepforest --config-name oam train --comet --tag maskrcnn --tag oam-tcd "$@"
