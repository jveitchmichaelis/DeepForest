#!/bin/bash
# One frozen LVD lidar pretrain: fold 0, 10 epochs, flip only.
# Follow with three unfrozen, flip-only NEON finetunes (seeds 0, 1, 2).
#
# Start from hf_model_last. Compare finetune epoch 19 manually and use
# hf_model_last for external evaluation. NEON test_cleaned is a
# development set, not an untouched test set.
#
# Run from the repository root:
#   bash src/deepforest/scripts/HPC/run_dinov3_lvd.sh
set -euo pipefail

HPC=src/deepforest/scripts/HPC
LIDAR=/blue/ewhite/veitchmichaelisj/deeplidar/output
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
LOGS=logs/dinov3_lidar/lvd_followup/$(date +%Y%m%d_%H%M%S)-$$
LVD=facebook/dinov3-vitl16-pretrain-lvd1689m

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# Some NEON and lidar boxes are a single pixel, which fails the coordinate check
NO_CHECK=(train.validate_coordinates=False validation.validate_coordinates=False)

# Pretrain on lidar fold 0 train, validate on lidar fold 0 val
LIDAR_DATA=("train.csv_file=$LIDAR/folds/all_annotations_train_fold_0.csv" "train.root_dir=$LIDAR"
            "validation.csv_file=$LIDAR/folds/all_annotations_val_fold_0.csv" "validation.root_dir=$LIDAR"
            "${NO_CHECK[@]}")

# Finetune on NEON train_cleaned; test_cleaned is used for development only.
NEON_DATA=("train.csv_file=$SPLITS/train_cleaned.csv" "train.root_dir=$NEON"
           "validation.csv_file=$SPLITS/test_cleaned.csv" "validation.root_dir=$NEON"
           "${NO_CHECK[@]}")

FLIP="train.augmentations=[HorizontalFlip]"

PRETRAIN_SOURCE=$LOGS/pretrain-lvd-frozen
job=$(PL_GLOBAL_SEED=0 sbatch --parsable --time=12:00:00 --mem=512GB "$HPC/dinov3_pretrain_b200.sbatch" \
  --tag lvd-followup --tag pretrain-lvd-frozen "log_root=$PRETRAIN_SOURCE" \
  "model.name=$LVD" train.freeze_backbone=True \
  train.epochs=10 train.scheduler.params.T_max=10 \
  "${LIDAR_DATA[@]}" "$FLIP")
echo "LVD pretrain job: $job; logs: $LOGS"

for seed in 0 1 2; do
  PRETRAIN_DIR=$PRETRAIN_SOURCE PRETRAIN_EXPORT=hf_model_last PL_GLOBAL_SEED=$seed \
    sbatch "--dependency=afterok:${job%%;*}" "$HPC/dinov3_finetune_b200.sbatch" \
    --tag lvd-followup --tag lvd "log_root=$LOGS/finetune-lvd-seed$seed" \
    train.freeze_backbone=False train.epochs=20 train.scheduler.params.T_max=20 \
    "${NEON_DATA[@]}" "$FLIP"
done
