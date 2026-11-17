#!/usr/bin/env bash
# Score MillionTrees test predictions per source for each named evaluation, in
# one CPU job, including MillionTrees' mask-aware precision. Reads logs/milliontrees/evaluation/NAME-mt-predictions.csv and
# writes NAME-mt-by-source.csv next to it.
#
#   bash src/deepforest/scripts/HPC/score_milliontrees.sh NAME [NAME...]
#
# For example: bash src/deepforest/scripts/HPC/score_milliontrees.sh mt-lvd-lidar mt-r50-pr7
# Run from the repository root.
set -euo pipefail

MT=${MT_ROOT:-/blue/ewhite/veitchmichaelisj/milliontrees/deepforest}
RELEASE=${RELEASE_DIR:-/blue/ewhite/veitchmichaelisj/milliontrees/download/TreeBoxes_supervised_v0.23}
EVAL=logs/milliontrees/evaluation

if [[ $# -eq 0 ]]; then
  echo "Usage: bash $0 NAME [NAME...]" >&2
  exit 2
fi

for name in "$@"; do
  if [[ ! -f "$EVAL/$name-mt-predictions.csv" ]]; then
    echo "Missing $EVAL/$name-mt-predictions.csv; is its evaluation finished?" >&2
    exit 2
  fi
done

srun --time=00:30:00 --cpus-per-task=16 --mem=32GB bash -c '
for name in "$@"; do
  echo "== $name"
  uv run python -m deepforest.scripts.score_milliontrees "'"$MT"'/test.csv" \
    "'"$EVAL"'/$name-mt-predictions.csv" \
    --official-csv "'"$RELEASE"'/within-distribution.csv" --masks-dir "'"$RELEASE"'/masks" \
    -o "'"$EVAL"'/$name-mt-by-source.csv"
done' scoring "$@"
