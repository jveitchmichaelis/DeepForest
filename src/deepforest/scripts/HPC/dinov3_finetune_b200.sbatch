#!/bin/bash
#SBATCH --job-name=dinov3-finetune
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=16
#SBATCH --mem=196GB
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

# Fine-tune with dinov3-finetune.yaml. Arguments are passed to
# `deepforest train`. Set PRETRAIN_DIR to the log_root of a pretrain job
# to start from its newest hf_model export, otherwise pass model.name.

printenv | grep -i slurm | sort

source .venv/bin/activate

export NCCL_NVLS_ENABLE=0
export COMET_PROJECT="${COMET_PROJECT:-lidar-pretrain-holdout}"
if [[ -f "$HOME/.secrets/hf_token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token")"
fi

if [[ -n "${PRETRAIN_DIR:-}" ]]; then
  # Newest hf_model export under the pretrain log_root
  HF_MODEL=$(find "$PRETRAIN_DIR" -type d -name hf_model -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -z "$HF_MODEL" ]]; then
    echo "No hf_model folder found under $PRETRAIN_DIR"
    exit 1
  fi
  echo "Fine-tuning from $HF_MODEL"
  set -- "model.name=$HF_MODEL" "$@"
fi

srun --kill-on-bad-exit=1 uv run deepforest --config-name=dinov3-finetune train --comet --export-hf "$@"
