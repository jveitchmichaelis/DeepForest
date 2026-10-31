#!/bin/bash
# Three SAT controls for the LVD followup, using the existing round-2
# 10-epoch, frozen, flip-only pretrain's last checkpoint. No new pretrain.
# Compare finetune epoch 19 manually in Comet.
#
# Run from the repository root:
#   bash src/deepforest/scripts/HPC/run_dinov3_sat_control.sh
set -euo pipefail

HPC=src/deepforest/scripts/HPC
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
SAT=logs/dinov3_lidar/round2/pretrain-ep10-flip/obliged_inventory_222/20261030_041701/hf_model_last
LOGS=logs/dinov3_lidar/lvd_followup/sat-$(date +%Y%m%d_%H%M%S)-$$

if [[ ! -f "$SAT/config.json" || ! -f "$SAT/model.safetensors" ]]; then
  echo "Missing SAT last-checkpoint export: $SAT" >&2
  exit 1
fi

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# NEON test_cleaned is used for development; report the final epoch.
echo "Submitting three SAT controls under $LOGS from $SAT"
for seed in 0 1 2; do
  PRETRAIN_DIR= PL_GLOBAL_SEED=$seed \
    sbatch --job-name="sat-control-seed$seed" "$HPC/dinov3_finetune_b200.sbatch" \
    --tag lvd-followup --tag sat-control "log_root=$LOGS/finetune-sat-seed$seed" \
    "model.name=$SAT" train.freeze_backbone=False \
    train.epochs=20 train.scheduler.params.T_max=20 \
    "train.csv_file=$SPLITS/train_cleaned.csv" "train.root_dir=$NEON" \
    "validation.csv_file=$SPLITS/test_cleaned.csv" "validation.root_dir=$NEON" \
    train.validate_coordinates=False validation.validate_coordinates=False \
    'train.augmentations=[HorizontalFlip]'
done
