#!/usr/bin/env bash
# The DINOv3 LVD + lidar model at native resolution throughout:
#   1. lidar pretrain on fold 0 (dinov3-pretrain-lvd-native.yaml)
#   2. MillionTrees finetune from its last epoch (milliontrees-dinov3.yaml)
#   3. evaluation of both on the NEON and MillionTrees test sets
# Each step starts only after the one before it succeeds.
#
#   bash src/deepforest/scripts/HPC/run_milliontrees_native.sh [--dry-run]
#
# Run from the repository root, after preparing MT with prepare_milliontrees.py.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
LIDAR=/blue/ewhite/veitchmichaelisj/deeplidar/output
MT=${MT_ROOT:-/blue/ewhite/veitchmichaelisj/milliontrees/deepforest}
PRETRAIN=lvd-lidar-native-pretrain
PRETRAIN_LOGS=logs/dinov3_lidar/lvd-lidar-native
PRETRAIN_EXPORT=$PRETRAIN_LOGS/$PRETRAIN/hf_model_last
FINETUNE=mt-lvd-lidar-native

export DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi

if [[ "$DRY_RUN" == false ]]; then
  for path in "$LIDAR/folds/all_annotations_train_fold_0.csv" "$LIDAR/folds/all_annotations_val_fold_0.csv" \
    "$MT/train.csv" "$MT/selection.csv" "$MT/test.csv"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing $path" >&2
      exit 2
    fi
  done
  if [[ -e "$PRETRAIN_LOGS/$PRETRAIN" || -e "logs/milliontrees/$FINETUNE" ]]; then
    echo "Output folders exist already; remove them or rename the runs" >&2
    exit 2
  fi
  mkdir -p slurm_logs
fi

# submit ARGS...: submit one sbatch job and print its job ID
submit() {
  if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' sbatch --parsable "$@" >&2
    printf '\n' >&2
    local arg
    for arg in "$@"; do [[ $arg == --job-name=* ]] && echo "<${arg#--job-name=}>"; done
    return 0
  fi
  sbatch --parsable "$@"
}

# The lidar validation keeps every prediction in memory, so it needs 512 GB
pretrain=$(COMET_PROJECT=dinov3-lidar submit \
  "--export=ALL,MODEL_CONFIG=dinov3-pretrain-lvd-native,PL_GLOBAL_SEED=0" \
  --job-name="$PRETRAIN" --time=12:00:00 --mem=512GB "$HPC/dinov3_pretrain_b200.sbatch" \
  --experiment-name "$PRETRAIN" --tag lvd-lidar-native "log_root=$PRETRAIN_LOGS" \
  "train.csv_file=$LIDAR/folds/all_annotations_train_fold_0.csv" "train.root_dir=$LIDAR" \
  "validation.csv_file=$LIDAR/folds/all_annotations_val_fold_0.csv" "validation.root_dir=$LIDAR")

finetune=$(COMET_PROJECT=milliontrees submit \
  "--export=ALL,MODEL_CONFIG=milliontrees-dinov3,PL_GLOBAL_SEED=0" \
  --dependency="afterok:$pretrain" --job-name="$FINETUNE" "$HPC/milliontrees_train_b200.sbatch" \
  --experiment-name "$FINETUNE" --tag milliontrees --tag lvd-lidar-native "model.name=$PRETRAIN_EXPORT" \
  "train.csv_file=$MT/train.csv" "train.root_dir=$MT/images" \
  "validation.csv_file=$MT/selection.csv" "validation.root_dir=$MT/images")

DEPENDENCY=$pretrain bash "$HPC/run_milliontrees_eval.sh" "$PRETRAIN" milliontrees-dinov3 \
  "model.name=$PRETRAIN_EXPORT"
DEPENDENCY=$finetune bash "$HPC/run_milliontrees_eval.sh" "$FINETUNE" milliontrees-dinov3 \
  "model.name=logs/milliontrees/$FINETUNE/hf_model_last"
