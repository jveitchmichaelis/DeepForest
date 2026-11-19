#!/bin/bash
# One LVD lidar pretrain (frozen by default): fold 0, 10 epochs, flip only.
# Follow with three unfrozen, flip-only NEON finetunes (seeds 0, 1, 2).
#
# Start from hf_model_last. Compare finetune epoch 19 manually and use
# hf_model_last for external evaluation. NEON test_cleaned is a
# development set, not an untouched test set.
#
# Run from the repository root:
#   bash src/deepforest/scripts/HPC/run_dinov3_lvd.sh
# Pass train.freeze_backbone=False to unfreeze the backbone during pretraining.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
LIDAR=/blue/ewhite/veitchmichaelisj/deeplidar/output
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
LOGS=logs/dinov3_lidar/lvd-lidar-followup

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# Pretrain on lidar fold 0 train, validate on lidar fold 0 val
LIDAR_DATA=("train.csv_file=$LIDAR/folds/all_annotations_train_fold_0.csv" "train.root_dir=$LIDAR"
            "validation.csv_file=$LIDAR/folds/all_annotations_val_fold_0.csv" "validation.root_dir=$LIDAR")

# Finetune on NEON train_cleaned; test_cleaned is used for development only.
NEON_DATA=("train.csv_file=$SPLITS/train_cleaned.csv" "train.root_dir=$NEON"
           "validation.csv_file=$SPLITS/test_cleaned.csv" "validation.root_dir=$NEON")

PRETRAIN_SOURCE=$LOGS/lvd-lidar-pretrain
job=$(MODEL_CONFIG=dinov3-pretrain-lvd PL_GLOBAL_SEED=0 \
  sbatch --parsable --time=12:00:00 --mem=512GB "$HPC/dinov3_pretrain_b200.sbatch" \
  --experiment-name lvd-lidar-pretrain --tag lvd-lidar-followup "log_root=$LOGS" \
  "${LIDAR_DATA[@]}" "$@")
echo "LVD pretrain job: $job; logs: $LOGS"

for seed in 0 1 2; do
  PRETRAIN_DIR=$PRETRAIN_SOURCE PRETRAIN_EXPORT=hf_model_last PL_GLOBAL_SEED=$seed \
    sbatch "--dependency=afterok:${job%%;*}" "$HPC/dinov3_finetune_b200.sbatch" \
    --experiment-name "lvd-neon-seed$seed" --tag lvd-lidar-followup "log_root=$LOGS" \
    "${NEON_DATA[@]}"
done
