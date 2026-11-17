#!/bin/bash
# Three historical SAT controls using the 10-epoch, frozen, flip-only
# lidar pretrain's last checkpoint. No new pretrain.
# Compare finetune epoch 19 manually in Comet.
#
# Run from the repository root:
#   bash src/deepforest/scripts/HPC/run_dinov3_sat_control.sh
set -euo pipefail

HPC=src/deepforest/scripts/HPC
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
# Last epoch of obliged_inventory_222 on Comet (dinov3-lidar)
SAT_LIDAR_EXPORT="${SAT_LIDAR_EXPORT:-/blue/ewhite/veitchmichaelisj/logs/dinov3_lidar/round2/pretrain-ep10-flip/obliged_inventory_222/20261030_041701/hf_model_last}"
LOGS=logs/dinov3_lidar/sat-lidar-control

if [[ ! -f "$SAT_LIDAR_EXPORT/config.json" || ! -f "$SAT_LIDAR_EXPORT/model.safetensors" ]]; then
  echo "Missing SAT lidar export: $SAT_LIDAR_EXPORT" >&2
  exit 1
fi

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# NEON test_cleaned is used for development; report the final epoch.
echo "Submitting three SAT controls under $LOGS from $SAT_LIDAR_EXPORT"
for seed in 0 1 2; do
  PRETRAIN_DIR= PL_GLOBAL_SEED=$seed \
    sbatch --job-name="sat-control-seed$seed" "$HPC/dinov3_finetune_b200.sbatch" \
    --experiment-name "sat-neon-seed$seed" --tag sat-lidar-control "log_root=$LOGS" \
    "model.name=$SAT_LIDAR_EXPORT" \
    "train.csv_file=$SPLITS/train_cleaned.csv" "train.root_dir=$NEON" \
    "validation.csv_file=$SPLITS/test_cleaned.csv" "validation.root_dir=$NEON"
done
