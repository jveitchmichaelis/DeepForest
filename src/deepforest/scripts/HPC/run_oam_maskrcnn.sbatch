#!/bin/bash
#SBATCH --job-name=maskrcnn-oam
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err
#SBATCH --ntasks-per-node=1

# Mask R-CNN on OAM-TCD. Build the dataset once with
# src/deepforest/scripts/HPC/prepare_tcd.py, which writes the data/tcd paths
# conf/oam.yaml points at.
#
#   sbatch src/deepforest/scripts/HPC/run_oam_maskrcnn.sbatch
#   sbatch src/deepforest/scripts/HPC/run_oam_maskrcnn.sbatch ++train.epochs=50

printenv | grep -i slurm | sort

export COMET_PROJECT="deepforest-maskrcnn"

mkdir -p slurm_logs

srun uv run deepforest --config-name oam train --comet --tag maskrcnn --tag oam-tcd "$@"

# Training validates on a 1024 centre crop
CKPT=$(ls -t logs/maskrcnn/*/*/checkpoints/last.ckpt | head -1)

# Then evaluate on 2048 tiles
srun uv run deepforest --config-name oam evaluate \
    --checkpoint "$CKPT" \
    -o "$(dirname "$CKPT")/full_tile_metrics.csv" \
    "validation.augmentations=[]"
