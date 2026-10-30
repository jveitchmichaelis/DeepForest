#!/bin/bash
# Compare flip only augmentation with stronger augmentation during the NEON
# finetune, for the DINOv3 RetinaNet and the release checkpoint.
#
# Each setting runs with three seeds. PL_GLOBAL_SEED sets the batch order,
# which is otherwise the same in every run.
#
# The DINOv3 runs start from the frozen pretrain made by the first round
# (logs/dinov3_lidar/pretrain-frozen).
#
# Run from the repository root.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
OUT=logs/dinov3_lidar/augment

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# Train on NEON train_cleaned, validate on NEON test_cleaned.
# Some NEON boxes are a single pixel, which fails the coordinate check.
NEON_DATA="train.csv_file=$SPLITS/train_cleaned.csv train.root_dir=$NEON
           validation.csv_file=$SPLITS/test_cleaned.csv validation.root_dir=$NEON
           train.validate_coordinates=False validation.validate_coordinates=False"

FLIP="train.augmentations=[HorizontalFlip]"
STRONG="train.augmentations=[HorizontalFlip,VerticalFlip,Rotate90,RandomScale,RandomBrightnessContrast,MotionBlur]"

# finetune NAME SEED AUGMENTATION [OVERRIDES...]
# Submits a NEON finetune logged under $OUT/NAME-seedSEED.
finetune() {
  local name=$1 seed=$2 aug=$3
  shift 3
  PL_GLOBAL_SEED=$seed sbatch $HPC/dinov3_finetune_b200.sbatch \
    --tag augment --tag $name log_root=$OUT/$name-seed$seed \
    $NEON_DATA "$aug" "$@"
}

# DINOv3: start from the frozen pretrain and train the whole model
export PRETRAIN_DIR=logs/dinov3_lidar/pretrain-frozen
DINOV3="train.freeze_backbone=False"

finetune dinov3-flip 0 "$FLIP" $DINOV3
finetune dinov3-flip 1 "$FLIP" $DINOV3
finetune dinov3-flip 2 "$FLIP" $DINOV3

finetune dinov3-strong 0 "$STRONG" $DINOV3
finetune dinov3-strong 1 "$STRONG" $DINOV3
finetune dinov3-strong 2 "$STRONG" $DINOV3

# Release checkpoint: a ResNet-50 RetinaNet, so no pretrain folder
unset PRETRAIN_DIR
RELEASE="architecture=retinanet model.name=weecology/deepforest-tree train.lr=0.02"

finetune release-flip 0 "$FLIP" $RELEASE
finetune release-flip 1 "$FLIP" $RELEASE
finetune release-flip 2 "$FLIP" $RELEASE

finetune release-strong 0 "$STRONG" $RELEASE
finetune release-strong 1 "$STRONG" $RELEASE
finetune release-strong 2 "$STRONG" $RELEASE
