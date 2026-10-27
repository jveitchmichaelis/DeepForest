#!/bin/bash
# Submit the DINOv3 lidar pretrain -> NEON finetune runs and the release
# checkpoint baselines. Run from the repository root.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
LIDAR=/blue/ewhite/veitchmichaelisj/deeplidar/output
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
LOGS=logs/dinov3_lidar

# Pretrain validates on the NEON training set
PRETRAIN_DATA="train.csv_file=$LIDAR/all_annotations.csv train.root_dir=$LIDAR validation.csv_file=$SPLITS/train_cleaned.csv validation.root_dir=$NEON"
# Next round: validate on a lidar fold so NEON is not used to pick the checkpoint
# PRETRAIN_DATA="train.csv_file=$LIDAR/folds/all_annotations_train_fold_0.csv train.root_dir=$LIDAR validation.csv_file=$LIDAR/folds/all_annotations_val_fold_0.csv validation.root_dir=$LIDAR"
# Some NEON and lidar boxes are a single pixel, which fails the coordinate check
NO_CHECK="train.validate_coordinates=False validation.validate_coordinates=False"
NEON_DATA="train.csv_file=$SPLITS/train_cleaned.csv train.root_dir=$NEON validation.csv_file=$SPLITS/test_cleaned.csv validation.root_dir=$NEON"

mkdir -p slurm_logs

# Frozen backbone: pretrain, then finetune
pretrain=$(sbatch --parsable $HPC/dinov3_pretrain_b200.sbatch --tag pretrain-frozen log_root=$LOGS/pretrain-frozen $PRETRAIN_DATA $NO_CHECK)
PRETRAIN_DIR=$LOGS/pretrain-frozen sbatch --dependency=afterok:$pretrain $HPC/dinov3_finetune_b200.sbatch --tag finetune-frozen log_root=$LOGS/finetune-frozen $NEON_DATA $NO_CHECK

# Unfrozen backbone
pretrain=$(sbatch --parsable $HPC/dinov3_pretrain_b200.sbatch --tag pretrain-unfrozen train.freeze_backbone=False log_root=$LOGS/pretrain-unfrozen $PRETRAIN_DATA $NO_CHECK)
PRETRAIN_DIR=$LOGS/pretrain-unfrozen sbatch --dependency=afterok:$pretrain $HPC/dinov3_finetune_b200.sbatch --tag finetune-unfrozen train.freeze_backbone=False log_root=$LOGS/finetune-unfrozen $NEON_DATA $NO_CHECK

# Release checkpoint: evaluate as is, and finetune with the same recipe
sbatch $HPC/evaluate_b200.sbatch $SPLITS/test_cleaned.csv --root-dir $NEON -o $LOGS/release/results.csv model.name=weecology/deepforest-tree batch_size=16 $NO_CHECK
sbatch $HPC/dinov3_finetune_b200.sbatch --tag finetune-release log_root=$LOGS/finetune-release architecture=retinanet model.name=weecology/deepforest-tree train.lr=0.02 $NEON_DATA $NO_CHECK
