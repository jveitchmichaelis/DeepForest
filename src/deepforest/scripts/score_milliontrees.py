"""Score saved predictions per MillionTrees source.

For each source in a test CSV prepared by prepare_milliontrees, compute
map_50, box recall and box precision, with the number of images and the
share of completely annotated images. Add three summary rows: the
unweighted mean over sources (macro average), all images together, and
completely annotated images only. On incompletely annotated images,
detections of unlabelled trees count as false positives, so precision and
mAP are underestimates there.

The metrics match DeepForest validation: map_50 counts up to
--max-detections detections per image, and box recall and precision use
--iou-threshold. The source is the first folder of each image path, as
written by prepare_milliontrees; images without a folder form one group.

Usage:
    python -m deepforest.scripts.score_milliontrees test.csv predictions.csv \\
        --official-csv TreeBoxes_supervised_v0.23/within-distribution.csv -o scores.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
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


def score(
    images: list[str],
    predictions: dict[str, dict],
    targets: dict[str, dict],
    iou_threshold: float,
    max_detections: int,
) -> dict:
    """map_50, map, box recall and box precision for a set of images."""
    mean_ap = MeanAveragePrecision(
        backend="faster_coco_eval",
        max_detection_thresholds=sorted({1, 10, max_detections}),
    )
    recall_precision = RecallPrecision(iou_threshold=iou_threshold, label_dict=LABELS)
    preds = [predictions[image] for image in images]
    truth = [targets[image] for image in images]
    mean_ap.update(preds, truth)
    recall_precision.update(preds, truth, images)
    result = {key: float(value) for key, value in recall_precision.compute().items()}
    ap = mean_ap.compute()
    result.update({"map_50": float(ap["map_50"]), "map": float(ap["map"])})
    return result


def score_by_source(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    complete: set[str] | None = None,
    iou_threshold: float = 0.4,
    max_detections: int = 2000,
) -> pd.DataFrame:
    """One row per source, then the macro average, all images and, when
    complete is given, completely annotated images only."""
    images = pd.Series(ground_truth.image_path.unique())
    sources = images.str.split("/").str[0].where(images.str.contains("/"), "")
    is_complete = (
        images.str.split("/").str[-1].isin(complete) if complete is not None else None
    )
    targets = to_targets(ground_truth, images, scores=False)
    preds = to_targets(predictions, images, scores=True)

    def row(group, selected):
        result = {"group": group, "images": int(selected.sum())}
        if is_complete is not None:
            result["complete_share"] = float(is_complete[selected].mean())
        return {
            **result,
            **score(
                images[selected].tolist(), preds, targets, iou_threshold, max_detections
            ),
        }

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
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--max-detections", type=int, default=2000)
    parser.add_argument("-o", "--output", type=Path, help="Write the scores to this CSV")
    args = parser.parse_args()

    complete = complete_images(args.official_csv) if args.official_csv else None
    scores = score_by_source(
        pd.read_csv(args.ground_truth),
        pd.read_csv(args.predictions),
        complete,
        args.iou_threshold,
        args.max_detections,
    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(scores.to_string(index=False))
    if args.output:
        scores.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
