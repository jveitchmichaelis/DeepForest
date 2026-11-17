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
NEON_DATA="train.csv_file=$SPLITS/train_cleaned.csv train.root_dir=$NEON
           validation.csv_file=$SPLITS/test_cleaned.csv validation.root_dir=$NEON"

FLIP="train.augmentations=[HorizontalFlip]"
STRONG="train.augmentations=[HorizontalFlip,VerticalFlip,Rotate90,RandomScale,RandomBrightnessContrast,MotionBlur]"

# finetune NAME SEED AUGMENTATION [OVERRIDES...]
# Submits a NEON finetune logged under $OUT/<comet_run_name>.
finetune() {
  local name=$1 seed=$2 aug=$3
  shift 3
  MODEL_CONFIG="${MODEL_CONFIG:-dinov3-finetune}" PL_GLOBAL_SEED=$seed \
    sbatch $HPC/dinov3_finetune_b200.sbatch \
    --experiment-name "$name-seed$seed" --tag augment --tag "$name" "log_root=$OUT" \
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
MODEL_CONFIG=resnet-finetune finetune release-flip 0 "$FLIP"
MODEL_CONFIG=resnet-finetune finetune release-flip 1 "$FLIP"
MODEL_CONFIG=resnet-finetune finetune release-flip 2 "$FLIP"

MODEL_CONFIG=resnet-finetune finetune release-strong 0 "$STRONG"
MODEL_CONFIG=resnet-finetune finetune release-strong 1 "$STRONG"
MODEL_CONFIG=resnet-finetune finetune release-strong 2 "$STRONG"
