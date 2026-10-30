#!/bin/bash
# Round 2 of the DINOv3 lidar pretrain experiments.
#
# Four pretrains on lidar fold 0, all with the DINOv3 encoder frozen:
#   - 2 or 10 epochs
#   - flip only or strong augmentation
# They validate on the lidar fold 0 validation set, so NEON is not used to
# pick the checkpoint.
#
# Each pretrain is followed by four NEON finetunes with the encoder unfrozen:
# flip only or strong augmentation, each with seeds 0 and 1. A finetune
# starts only if its pretrain finished without error.
#
# Run from the repository root.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
LIDAR=/blue/ewhite/veitchmichaelisj/deeplidar/output
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
LOGS=logs/dinov3_lidar/round2

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# Some NEON and lidar boxes are a single pixel, which fails the coordinate check
NO_CHECK="train.validate_coordinates=False validation.validate_coordinates=False"

# Pretrain on lidar fold 0 train, validate on lidar fold 0 val
LIDAR_DATA="train.csv_file=$LIDAR/folds/all_annotations_train_fold_0.csv train.root_dir=$LIDAR
            validation.csv_file=$LIDAR/folds/all_annotations_val_fold_0.csv validation.root_dir=$LIDAR
            $NO_CHECK"

# Finetune on NEON train_cleaned, validate on NEON test_cleaned
NEON_DATA="train.csv_file=$SPLITS/train_cleaned.csv train.root_dir=$NEON
           validation.csv_file=$SPLITS/test_cleaned.csv validation.root_dir=$NEON
           $NO_CHECK"

FLIP="train.augmentations=[HorizontalFlip]"
STRONG="train.augmentations=[HorizontalFlip,VerticalFlip,Rotate90,RandomScale,RandomBrightnessContrast,MotionBlur]"

# pretrain NAME EPOCHS TIME AUGMENTATION
# Submits a frozen encoder pretrain and prints its job ID. The cosine
# schedule length (T_max) is set to the number of epochs.
# Validation on the lidar fold keeps every prediction in memory on each
# rank, which needs more than the default 196GB.
pretrain() {
  local name=$1 epochs=$2 time=$3 aug=$4
  sbatch --parsable --time=$time --mem=512GB $HPC/dinov3_pretrain_b200.sbatch \
    --tag round2 --tag $name log_root=$LOGS/$name \
    train.epochs=$epochs train.scheduler.params.T_max=$epochs \
    $LIDAR_DATA "$aug"
}

# finetune PRETRAIN_JOB PRETRAIN_NAME NAME SEED AUGMENTATION
# Submits an unfrozen NEON finetune that starts from the pretrain's best
# export once the pretrain job has finished.
finetune() {
  local job=$1 pretrain_name=$2 name=$3 seed=$4 aug=$5
  PRETRAIN_DIR=$LOGS/$pretrain_name PL_GLOBAL_SEED=$seed \
    sbatch --dependency=afterok:$job $HPC/dinov3_finetune_b200.sbatch \
    --tag round2 --tag $name log_root=$LOGS/$name-seed$seed \
    train.freeze_backbone=False $NEON_DATA "$aug"
}

# 20 epochs on all lidar tiles took 11-13 h, so these time limits leave room.

# Pretrain: 2 epochs, flip only
job=$(pretrain pretrain-ep2-flip 2 04:00:00 "$FLIP")
finetune $job pretrain-ep2-flip finetune-ep2-flip-ftflip   0 "$FLIP"
finetune $job pretrain-ep2-flip finetune-ep2-flip-ftflip   1 "$FLIP"
finetune $job pretrain-ep2-flip finetune-ep2-flip-ftstrong 0 "$STRONG"
finetune $job pretrain-ep2-flip finetune-ep2-flip-ftstrong 1 "$STRONG"

# Pretrain: 2 epochs, strong augmentation
job=$(pretrain pretrain-ep2-strong 2 04:00:00 "$STRONG")
finetune $job pretrain-ep2-strong finetune-ep2-strong-ftflip   0 "$FLIP"
finetune $job pretrain-ep2-strong finetune-ep2-strong-ftflip   1 "$FLIP"
finetune $job pretrain-ep2-strong finetune-ep2-strong-ftstrong 0 "$STRONG"
finetune $job pretrain-ep2-strong finetune-ep2-strong-ftstrong 1 "$STRONG"

# Pretrain: 10 epochs, flip only
job=$(pretrain pretrain-ep10-flip 10 10:00:00 "$FLIP")
finetune $job pretrain-ep10-flip finetune-ep10-flip-ftflip   0 "$FLIP"
finetune $job pretrain-ep10-flip finetune-ep10-flip-ftflip   1 "$FLIP"
finetune $job pretrain-ep10-flip finetune-ep10-flip-ftstrong 0 "$STRONG"
finetune $job pretrain-ep10-flip finetune-ep10-flip-ftstrong 1 "$STRONG"

# Pretrain: 10 epochs, strong augmentation
job=$(pretrain pretrain-ep10-strong 10 10:00:00 "$STRONG")
finetune $job pretrain-ep10-strong finetune-ep10-strong-ftflip   0 "$FLIP"
finetune $job pretrain-ep10-strong finetune-ep10-strong-ftflip   1 "$FLIP"
finetune $job pretrain-ep10-strong finetune-ep10-strong-ftstrong 0 "$STRONG"
finetune $job pretrain-ep10-strong finetune-ep10-strong-ftstrong 1 "$STRONG"
