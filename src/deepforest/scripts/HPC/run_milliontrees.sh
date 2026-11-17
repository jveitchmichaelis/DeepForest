#!/usr/bin/env bash
# Finetune the DINOv3 LVD + lidar RetinaNet on MillionTrees (MT) with
# milliontrees-dinov3.yaml, then evaluate it on the NEON and MT test sets.
# Also evaluates two references: the LVD + lidar pretrain before the MT
# finetune, and the release model (as trained, and at native resolution).
#
#   bash src/deepforest/scripts/HPC/run_milliontrees.sh [--dry-run]
#
# Run from the repository root, after preparing MT with prepare_milliontrees.py.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
MT=${MT_ROOT:-/blue/ewhite/veitchmichaelisj/milliontrees/deepforest}
LOGS=logs/milliontrees
NAME=mt-lvd-lidar
# Last epoch of steady_archway_4101 on Comet (dinov3-lidar)
LVD_LIDAR=/blue/ewhite/veitchmichaelisj/logs/dinov3_lidar/lvd_followup/20261030_164618-360387/pretrain-lvd-frozen/steady_archway_4101/20261030_174000/hf_model_last

export DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi

if [[ "$DRY_RUN" == false ]]; then
  for path in "$MT/train.csv" "$MT/selection.csv" "$MT/test.csv" "$LVD_LIDAR/model.safetensors"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing $path" >&2
      exit 2
    fi
  done
  mkdir -p slurm_logs
fi

export COMET_PROJECT=milliontrees

train=(sbatch --parsable "--export=ALL,MODEL_CONFIG=milliontrees-dinov3,PL_GLOBAL_SEED=0"
  --job-name="$NAME" "$HPC/milliontrees_train_b200.sbatch"
  --experiment-name "$NAME" --tag milliontrees --tag lvd-lidar "model.name=$LVD_LIDAR"
  "train.csv_file=$MT/train.csv" "train.root_dir=$MT/images"
  "validation.csv_file=$MT/selection.csv" "validation.root_dir=$MT/images")
if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' "${train[@]}"
  printf '\n'
  job="<$NAME>"
else
  job=$("${train[@]}")
fi

DEPENDENCY=$job bash "$HPC/run_milliontrees_eval.sh" "$NAME" milliontrees-dinov3 \
  "model.name=$LOGS/$NAME/hf_model_last"
bash "$HPC/run_milliontrees_eval.sh" lvd-lidar-pretrain milliontrees-dinov3 "model.name=$LVD_LIDAR"
bash "$HPC/run_milliontrees_eval.sh" release resnet-finetune
bash "$HPC/run_milliontrees_eval.sh" release-native resnet-finetune model.resize=False
