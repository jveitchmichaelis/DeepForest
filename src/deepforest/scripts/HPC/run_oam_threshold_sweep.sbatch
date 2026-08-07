#!/bin/bash
#SBATCH --job-name=maskrcnn-oam-sweep
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --output=./slurm_logs/%A.out
#SBATCH --error=./slurm_logs/%A.err
#SBATCH --ntasks-per-node=1

# Score a trained checkpoint on full tiles under different inference-time
# settings. These are constructor arguments, so no retraining is needed.
# oam.yaml uses score_thresh 0.2 and RPN test top-n 512; torchvision and
# Detectron2 both default to 0.05 and 1000. score_thresh filters before NMS,
# so it caps recall and therefore AP.
#
#   sbatch src/deepforest/scripts/HPC/run_oam_threshold_sweep.sbatch
#   sbatch src/deepforest/scripts/HPC/run_oam_threshold_sweep.sbatch path/to/last.ckpt
#
# Each pass over the 439 test tiles takes ~40 min.

printenv | grep -i slurm | sort

export COMET_PROJECT="deepforest-maskrcnn"

mkdir -p slurm_logs

CKPT=${1:-$(ls -t logs/maskrcnn/*/*/checkpoints/last.ckpt | head -1)}
OUT=$(dirname "$CKPT")/sweep
mkdir -p "$OUT"

echo "Checkpoint: $CKPT"
echo "Output:     $OUT"

evaluate () {
    name=$1
    shift
    echo "=== $name: $* ==="
    srun uv run deepforest --config-name oam evaluate \
        --checkpoint "$CKPT" \
        -o "$OUT/$name.csv" \
        "validation.augmentations=[]" \
        "$@"
}

evaluate baseline
evaluate score005 ++score_thresh=0.05
evaluate rpn1000 ++maskrcnn.rpn_pre_nms_top_n_test=1000 ++maskrcnn.rpn_post_nms_top_n_test=1000
evaluate score005_rpn1000 \
    ++score_thresh=0.05 \
    ++maskrcnn.rpn_pre_nms_top_n_test=1000 \
    ++maskrcnn.rpn_post_nms_top_n_test=1000

# detections_per_img also sets the metric's maxDets, so this one is a ceiling
# check rather than a number comparable to the Detectron2 baseline at 256.
evaluate score005_dets512 ++score_thresh=0.05 ++detections_per_img=512

echo "=== summary ==="
for f in "$OUT"/*.csv; do
    echo "--- $(basename "$f" .csv)"
    grep -E "^(map|map_50|map_tree|map_canopy|mar_256|polygon_recall)," "$f"
done

# One experiment holding every variant, as <variant>_<metric>.
srun uv run python src/deepforest/scripts/HPC/log_sweep_to_comet.py \
    --sweep-dir "$OUT" \
    --experiment-name "oam-threshold-sweep-${SLURM_JOB_ID}" \
    --tag sweep \
    --tag oam-tcd \
    --checkpoint "$CKPT"
