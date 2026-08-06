import os
from collections.abc import Callable
from warnings import warn

import pandas as pd
from omegaconf import DictConfig

from deepforest import distributed
from deepforest.main import deepforest

# Metric groups for a readable report. Each key is matched once, in order; keys
# matching no group fall through to an "Other" section so nothing is dropped.
_METRIC_GROUPS: list[tuple[str, Callable[[str], bool]]] = [
    ("Mean average precision", lambda k: k == "map" or k.startswith("map_")),
    ("Mean average recall", lambda k: k == "mar" or k.startswith("mar_")),
    (
        "Recall / precision",
        lambda k: (
            k.lower().endswith(("_recall", "_precision")) or k == "empty_frame_accuracy"
        ),
    ),
    ("Losses", lambda k: k.startswith("val_")),
]


def _report_metrics(m: deepforest, metrics: dict[str, float]) -> None:
    """Print validation metrics grouped into readable sections."""
    m.print("Evaluation Results:")
    reported: set[str] = set()
    for title, predicate in _METRIC_GROUPS:
        rows = {k: v for k, v in metrics.items() if k not in reported and predicate(k)}
        if not rows:
            continue
        reported.update(rows)
        m.print(f"\n{title}:")
        for k in sorted(rows):
            m.print(f"  {k:28s} {rows[k]:.4f}")

    leftover = {k: v for k, v in metrics.items() if k not in reported}
    if leftover:
        m.print("\nOther:")
        for k in sorted(leftover):
            m.print(f"  {k:28s} {leftover[k]:.4f}")


def evaluate(
    config: DictConfig,
    ground_truth: str | None = None,
    root_dir: str | None = None,
    output_path: str | None = None,
    save_predictions: str | None = None,
    checkpoint: str | None = None,
) -> None:
    """Run the validation loop on ground truth annotations and report metrics.

    Runs ``trainer.validate`` directly, so the reported numbers are exactly the
    training-time validation metrics: torchmetrics mAP (segmentation mAP for
    polygon models, bounding-box mAP otherwise) plus recall/precision.

    Examples:

    1. Evaluate the pretrained release weights on a ground-truth CSV::

        deepforest evaluate ground_truth.csv --root-dir /path/to/images

    2. Evaluate a trained checkpoint, saving generated predictions and a metrics
       summary::

        deepforest evaluate ground_truth.csv --root-dir /path/to/images \\
            --checkpoint model.ckpt \\
            --save-predictions predictions.csv -o eval_results.csv

    Args:
        config (DictConfig): DeepForest configuration.
        ground_truth (Optional[str]): Path to ground truth CSV file with annotations. If None, uses config.validation.csv_file.
        root_dir (Optional[str]): Root directory containing images. If None, uses config value or directory of csv_file.
        output_path (Optional[str]): Path to save evaluation metrics summary CSV.
        save_predictions (Optional[str]): Path to save generated predictions CSV.
        checkpoint (Optional[str]): Path to a model checkpoint to load weights from. The
            passed ``config`` (Hydra-composed) supplies evaluation settings, so it must be
            architecture-compatible with the checkpoint. If None, a fresh model is built
            from ``config`` (e.g. the pretrained release weights).

    Returns:
        None
    """
    if checkpoint is not None:
        m = deepforest.load_from_checkpoint(checkpoint, config=config)
    else:
        m = deepforest(config=config)

    if ground_truth is None:
        if config.validation.csv_file is None:
            raise ValueError(
                "No CSV file provided and config.validation.csv_file is not set"
            )
        ground_truth = config.validation.csv_file
        m.print(f"Using validation CSV from config: {ground_truth}")

    m.config.validation.csv_file = ground_truth
    if root_dir is not None:
        m.config.validation.root_dir = root_dir

    # Force metric computation on this single pass (default interval may skip it).
    m.config.validation.val_accuracy_interval = 1
    m.create_trainer()
    m.trainer.validate(m)

    metrics = {k: float(v) for k, v in m.trainer.logged_metrics.items()}

    # Save generated predictions if requested. collect_predictions gathers
    # across ranks, so every rank must call it; only rank zero writes.
    predictions_df = m.collect_predictions() if save_predictions is not None else None

    if not distributed.is_global_zero(m.trainer):
        return

    _report_metrics(m, metrics)

    if predictions_df is not None:
        if not predictions_df.empty:
            if os.path.dirname(save_predictions):
                os.makedirs(os.path.dirname(save_predictions), exist_ok=True)
            predictions_df.to_csv(save_predictions, index=False)
            m.print(f"Generated predictions saved to: {save_predictions}")
        else:
            warn(
                "Warning: No predictions to save (predictions dataframe is empty)",
                stacklevel=2,
            )

    # Save metrics summary to CSV if requested.
    if output_path is not None:
        summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        m.print(f"Evaluation results saved to: {output_path}")
