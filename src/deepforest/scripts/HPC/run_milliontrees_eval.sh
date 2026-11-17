#!/usr/bin/env bash
# Evaluate one model on whole images of the NEON and MillionTrees (MT) test
# sets. Submits two milliontrees_eval_b200.sbatch jobs, one image per batch so
# that images keep their size when the config sets model.resize: False.
# Every evaluation counts up to 2000 detections per image, so scores compare.
#
#   bash src/deepforest/scripts/HPC/run_milliontrees_eval.sh NAME CONFIG [OVERRIDES...]
#
# Metrics go to logs/milliontrees/evaluation/NAME-{neon,mt}.csv and predictions
# to NAME-{neon,mt}-predictions.csv, for per-source scores. Set DEPENDENCY
# to a job ID to start after that job succeeds, and DRY_RUN=true to print the
# commands. Run from the repository root.
set -euo pipefail

HPC=src/deepforest/scripts/HPC
SPLITS=/home/veitchmichaelisj/code/milliontrees_transformers/data/cv_splits
NEON=/orange/ewhite/DeepForest/NEON_benchmark/images
MT=${MT_ROOT:-/blue/ewhite/veitchmichaelisj/milliontrees/deepforest}
OUT=logs/milliontrees/evaluation

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 NAME CONFIG [OVERRIDES...]" >&2
  exit 2
fi
name=$1 config=$2
shift 2

[[ "${DRY_RUN:-false}" == true ]] || mkdir -p slurm_logs "$OUT"

# evaluate TEST CSV IMAGES [OVERRIDES...]: submit one evaluation
evaluate() {
  local test=$1 csv=$2 images=$3
  shift 3
  local -a command=(sbatch --parsable "--export=ALL,MODEL_CONFIG=$config"
    ${DEPENDENCY:+"--dependency=afterok:$DEPENDENCY"} --job-name="eval-$name-$test"
    "$HPC/milliontrees_eval_b200.sbatch" "$csv" --root-dir "$images" -o "$OUT/$name-$test.csv"
    --save-predictions "$OUT/$name-$test-predictions.csv"
    batch_size=1 "validation.augmentations=[]" detections_per_img=2000 topk_candidates=2000
    "$@")
  if [[ "${DRY_RUN:-false}" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

evaluate neon "$SPLITS/test_cleaned.csv" "$NEON" "$@"
evaluate mt "$MT/test.csv" "$MT/images" "$@"
