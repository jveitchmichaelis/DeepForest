#!/bin/bash
# Submit NEON finetune ablations for the DINOv3 RetinaNet. Uses the pretrain
# exports from run_dinov3_pretrain.sh, so run that first. Run from the
# repository root.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
LOGS=logs/dinov3_lidar
OUT=$LOGS/ablations
FROZEN_PT=$LOGS/pretrain-frozen
UNFROZEN_PT=$LOGS/pretrain-unfrozen
SAT=facebook/dinov3-vitl16-pretrain-sat493m
LVD=facebook/dinov3-vitl16-pretrain-lvd1689m

# Some NEON boxes are a single pixel, which fails the coordinate check
NO_CHECK="train.validate_coordinates=False validation.validate_coordinates=False"
NEON_DATA="train.csv_file=$SPLITS/train_cleaned.csv train.root_dir=$NEON validation.csv_file=$SPLITS/test_cleaned.csv validation.root_dir=$NEON $NO_CHECK"
FROZEN="train.freeze_backbone=True"
UNFROZEN="train.freeze_backbone=False"

export COMET_PROJECT=dinov3-lidar
mkdir -p slurm_logs

# finetune <name> <overrides...>: finetune on NEON, logged under $OUT/<name>
finetune() {
  local name=$1
  shift
  sbatch "$HPC/dinov3_finetune_b200.sbatch" --tag ablation --tag "$name" log_root=$OUT/$name $NEON_DATA "$@"
}

# 1. No lidar pretrain: encoder straight to NEON, for both encoders
finetune nopt-sat-frozen model.name=$SAT $FROZEN
finetune nopt-sat-unfrozen model.name=$SAT $UNFROZEN
finetune nopt-lvd-frozen model.name=$LVD $FROZEN
finetune nopt-lvd-unfrozen model.name=$LVD $UNFROZEN

# 2. Start from the last pretrain epoch instead of the lowest NEON val_loss.
# The first pretrain round only exported the best epoch, so export last.ckpt.
for pt in frozen unfrozen; do
  pt_dir=$LOGS/pretrain-$pt
  last=$(find "$pt_dir" -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -z "$last" ]]; then
    echo "No last.ckpt found under $pt_dir"
    exit 1
  fi
  export_job=$(sbatch --parsable "$HPC/export_b200.sbatch" "$last" -o "$(dirname "$(dirname "$last")")/hf_model_last")
  freeze=$([[ $pt == frozen ]] && echo "$FROZEN" || echo "$UNFROZEN")
  PRETRAIN_DIR=$pt_dir PRETRAIN_EXPORT=hf_model_last sbatch --dependency=afterok:$export_job \
    "$HPC/dinov3_finetune_b200.sbatch" --tag ablation --tag "last-$pt" log_root=$OUT/last-$pt $NEON_DATA $freeze
done

# 3. Mixed: freeze during pretrain only, or during finetune only
PRETRAIN_DIR=$FROZEN_PT finetune pt-frozen-ft-unfrozen $UNFROZEN
PRETRAIN_DIR=$UNFROZEN_PT finetune pt-unfrozen-ft-frozen $FROZEN

# 4. Repeats of the two main finetunes, to measure run to run variation
for rep in 2 3; do
  PRETRAIN_DIR=$FROZEN_PT finetune frozen-rep$rep $FROZEN
  PRETRAIN_DIR=$UNFROZEN_PT finetune unfrozen-rep$rep $UNFROZEN
done

# 5. Lower learning rate for the unfrozen finetune (default 0.04)
PRETRAIN_DIR=$UNFROZEN_PT finetune unfrozen-lr0.01 $UNFROZEN train.lr=0.01

# 6. Release checkpoint (ResNet-50): evaluate without finetuning, and repeat
# the finetune so its spread can be compared with the DINOv3 repeats
RELEASE="architecture=retinanet model.name=weecology/deepforest-tree train.lr=0.02"
sbatch "$HPC/evaluate_b200.sbatch" $SPLITS/test_cleaned.csv --root-dir $NEON -o $OUT/release-eval/results.csv \
  model.name=weecology/deepforest-tree batch_size=16 $NO_CHECK
for rep in 2 3; do
  finetune release-rep$rep $RELEASE
done
