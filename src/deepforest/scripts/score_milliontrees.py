"""Score saved predictions per MillionTrees source.

For each source in a test CSV prepared by prepare_milliontrees, compute
map_50, box recall and box precision, with the number of images and the
share of completely annotated images. Add three summary rows: the
unweighted mean over sources (macro average), all images together, and
completely annotated images only. On incompletely annotated images,
detections of unlabelled trees count as false positives, so box precision
and mAP are underestimates there.

With --masks-dir (the masks folder of the official release), also compute
MillionTrees' mask-aware precision: an unmatched prediction is not a false
positive when at least half of its box covers tree pixels in the image's
tree coverage mask. It uses MillionTrees' own MaskAwareDetectionPrecision,
so the milliontrees package must be installed.

The metrics match DeepForest validation: map_50 counts up to
--max-detections detections per image, and box recall and precision use
--iou-threshold. The source is the first folder of each image path, as
written by prepare_milliontrees; images without a folder form one group.

Usage:
    python -m deepforest.scripts.score_milliontrees test.csv predictions.csv \\
        --official-csv TreeBoxes_supervised_v0.23/within-distribution.csv -o scores.csv
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

from deepforest.metrics import RecallPrecision

BOXES = ["xmin", "ymin", "xmax", "ymax"]
LABELS = {"Tree": 0}


def complete_images(official_csv: Path) -> set[str]:
    """File names that the official release marks as completely annotated.

    The complete column holds True, False or the text "Image is
    completely annotated". An image counts as complete when any of its
    rows says so.
    """
    official = pd.read_csv(
        official_csv, usecols=["filename", "complete"], low_memory=False
    )
    flag = official.complete.astype(str)
    complete = flag.str.lower().eq("true") | flag.eq("Image is completely annotated")
    return set(official.filename[complete])


def to_targets(frame: pd.DataFrame, image_paths, scores: bool) -> dict[str, dict]:
    """Box, label (and score) tensors per image; images without rows get empty
    tensors."""
    grouped = dict(list(frame.groupby("image_path")))
    empty = pd.DataFrame(columns=[*BOXES, "label", "score"])
    out = {}
    for path in image_paths:
        rows = grouped.get(path, empty)
        item = {
            "boxes": torch.tensor(rows[BOXES].to_numpy(dtype="float32")).reshape(-1, 4),
            "labels": torch.tensor(rows.label.map(LABELS).to_numpy(dtype="int64")),
        }
        if scores:
            item["scores"] = torch.tensor(rows.score.to_numpy(dtype="float32"))
        out[path] = item
    return out


def image_metrics(item: tuple) -> tuple[float, float, float]:
    """Box precision, box recall and mask-aware precision of one image.

    Box precision and recall follow RecallPrecision: precision is NaN when
    the image has no predictions and recall is NaN when it has no ground
    truth, so nanmean leaves those images out as RecallPrecision does.
    Mask-aware precision follows MillionTrees and is NaN without a mask.
    """
    pred, target, iou_threshold, mask_path = item
    metric = RecallPrecision(iou_threshold=iou_threshold, label_dict=LABELS)
    metric.update([pred], [target])
    precision = (
        float(metric.precision) if metric.num_images_with_predictions else float("nan")
    )
    recall = float(metric.recall) if metric.num_images_with_ground_truth else float("nan")

    maskaware = float("nan")
    if mask_path is not None:
        from milliontrees.common.metrics.all_metrics import MaskAwareDetectionPrecision

        mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L")) > 0)
        # One image per call, so the average that compute() returns is its value
        metric = MaskAwareDetectionPrecision(
            iou_threshold=iou_threshold, score_threshold=0.0
        )
        result = metric.compute([pred], [{**target, "tree_coverage_mask": mask}])
        maskaware = float(result[metric.agg_metric_field])
    return precision, recall, maskaware


def score_by_source(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    complete: set[str] | None = None,
    iou_threshold: float = 0.4,
    max_detections: int = 2000,
    workers: int = 1,
    masks_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per source, then the macro average, all images and, when
    complete is given, completely annotated images only.

    Box matching is the slow step, so it runs once per image, over
    workers processes, and each group averages the per-image values.
    """
    images = pd.Series(ground_truth.image_path.unique())
    sources = images.str.split("/").str[0].where(images.str.contains("/"), "")
    is_complete = (
        images.str.split("/").str[-1].isin(complete) if complete is not None else None
    )
    targets = to_targets(ground_truth, images, scores=False)
    preds = to_targets(predictions, images, scores=True)

    # Masks are named by the official image file name with a .png suffix
    mask_paths = (
        [masks_dir / f"{Path(image).stem}.png" for image in images]
        if masks_dir is not None
        else [None] * len(images)
    )
    missing = [path for path in mask_paths if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing, e.g. {missing[0]}")

    items = (
        (preds[image], targets[image], iou_threshold, mask)
        for image, mask in zip(images, mask_paths, strict=True)
    )
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            values = list(pool.map(image_metrics, items, chunksize=16))
    else:
        values = list(map(image_metrics, items))
    per_image = pd.DataFrame(
        values, columns=["box_precision", "box_recall", "maskaware_precision"]
    )
    if masks_dir is None:
        per_image = per_image.drop(columns="maskaware_precision")

    def row(group, selected):
        result = {"group": group, "images": int(selected.sum())}
        if is_complete is not None:
            result["complete_share"] = float(is_complete[selected].mean())
        result.update(per_image[selected.to_numpy()].mean().to_dict())
        mean_ap = MeanAveragePrecision(
            backend="faster_coco_eval",
            max_detection_thresholds=sorted({1, 10, max_detections}),
        )
        mean_ap.update(
            [preds[image] for image in images[selected]],
            [targets[image] for image in images[selected]],
        )
        ap = mean_ap.compute()
        return {**result, "map_50": float(ap["map_50"]), "map": float(ap["map"])}

    rows = [row(source, sources.eq(source)) for source in sorted(sources.unique())]
    per_source = pd.DataFrame(rows)
    macro = per_source.drop(columns=["group", "images"]).mean().to_dict()
    rows.append(
        {"group": "macro average", "images": int(per_source.images.sum()), **macro}
    )
    rows.append(row("all images", images.notna()))
    if is_complete is not None and is_complete.any():
        rows.append(row("complete images", is_complete))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ground_truth", type=Path, help="Prepared test CSV")
    parser.add_argument(
        "predictions", type=Path, help="CSV from deepforest evaluate --save-predictions"
    )
    parser.add_argument(
        "--official-csv", type=Path, help="Official release CSV with the complete column"
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="masks folder of the official release, for mask-aware precision",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--max-detections", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        help="Processes for box matching; defaults to the CPUs available to the job",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write the scores to this CSV")
    args = parser.parse_args()

    complete = complete_images(args.official_csv) if args.official_csv else None
    scores = score_by_source(
        pd.read_csv(args.ground_truth),
        pd.read_csv(args.predictions),
        complete,
        args.iou_threshold,
        args.max_detections,
        args.workers,
        args.masks_dir,
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(scores.to_string(index=False))
    if args.output:
        scores.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
