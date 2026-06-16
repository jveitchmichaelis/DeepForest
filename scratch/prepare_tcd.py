"""Export restor/tcd (OAM-TCD) to COCO JSON + image files for Mask R-CNN training.

Usage:
    # Full dataset (default out_dir is data/tcd, matching oam.yaml)
    uv run python scratch/prepare_tcd.py

    # Subset for local testing
    uv run python scratch/prepare_tcd.py --n_train 200 --n_test 50

    # Custom HuggingFace cache location (e.g. to avoid re-downloading)
    uv run python scratch/prepare_tcd.py --hf_cache_dir /path/to/hf_cache

The script saves images as .tif files and writes standard COCO JSON annotation
files (images + annotations + categories). These can be passed directly to
DeepForest via read_file() / PolygonDataset.
"""

import argparse
import json
import os

from datasets import load_dataset


def clamp_segmentation(segmentation, width: int, height: int):
    """Clamp polygon vertices to image bounds and drop degenerate polygons.

    Args:
        segmentation: COCO segmentation field — either a list of flat
            coordinate lists ``[[x1,y1,x2,y2,...], ...]`` or an RLE dict.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Clamped segmentation in the same format, or None if all polygons
        became degenerate after clamping.
    """
    if not isinstance(segmentation, list):
        return segmentation  # RLE — already a raster mask, nothing to clamp

    clamped = []
    for poly in segmentation:
        xs = [min(max(poly[i], 0), width) for i in range(0, len(poly), 2)]
        ys = [min(max(poly[i], 0), height) for i in range(1, len(poly), 2)]
        flat = [v for pair in zip(xs, ys) for v in pair]
        if len(set(zip(xs, ys))) >= 3:
            clamped.append(flat)

    return clamped if clamped else None


def clamp_bbox(bbox, width: int, height: int):
    """Clamp a COCO bbox [x, y, w, h] to image bounds.

    Returns:
        Clamped [x, y, w, h], or None if the box has no area after clamping.
    """
    x, y, w, h = bbox
    x1 = min(max(x, 0), width)
    y1 = min(max(y, 0), height)
    x2 = min(max(x + w, 0), width)
    y2 = min(max(y + h, 0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


CATEGORIES = [
    {"id": 1, "name": "tree", "supercategory": "tree"},
    {"id": 2, "name": "canopy", "supercategory": "tree"},
]


def export_split(ds_split, out_dir: str, n: int | None, split_name: str, image_ids: list[int] | None = None):
    img_dir = os.path.join(out_dir, split_name, "images")
    os.makedirs(img_dir, exist_ok=True)

    images_meta = []
    all_annotations = []
    ann_id_offset = 0

    if image_ids is not None:
        id_set = set(image_ids)
        ds_split = ds_split.filter(
            lambda x: int(x["image_id"]) in id_set,
            input_columns=["image_id"],
        )
        n_total = len(ds_split)
    else:
        n_total = len(ds_split) if n is None else min(n, len(ds_split))
        ds_split = ds_split.select(range(n_total))
    rows = iter(ds_split)

    print(f"Exporting {n_total} images from '{split_name}' split...")

    for row in rows:
        image_id = int(row["image_id"])
        height = int(row["height"])
        width = int(row["width"])

        # Save image
        fname = f"{image_id}.tif"
        fpath = os.path.join(img_dir, fname)
        if not os.path.exists(fpath) or image_ids is not None:
            row["image"].save(fpath)

        images_meta.append({"id": image_id, "file_name": fname, "height": height, "width": width})

        raw_anns = json.loads(row["coco_annotations"])
        for ann in raw_anns:
            seg = clamp_segmentation(ann["segmentation"], width, height)
            if seg is None:
                continue
            bbox = clamp_bbox(ann["bbox"], width, height)
            if bbox is None:
                continue
            ann_id_offset += 1
            entry = {
                "id": ann_id_offset,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "segmentation": seg,
                "area": ann.get("area", 0),
                "bbox": bbox,
                "iscrowd": 0,
            }
            all_annotations.append(entry)

        if len(images_meta) % 100 == 0:
            print(f"  {len(images_meta)}/{n_total}")

    coco_json = {
        "images": images_meta,
        "annotations": all_annotations,
        "categories": CATEGORIES,
    }

    json_path = os.path.join(out_dir, f"{split_name}.json")
    with open(json_path, "w") as f:
        json.dump(coco_json, f)

    print(f"Wrote {len(images_meta)} images, {len(all_annotations)} annotations → {json_path}")
    return json_path, img_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/tcd",
                        help="Output directory for images and COCO JSON files.")
    parser.add_argument("--hf_cache_dir", default=None,
                        help="HuggingFace dataset cache directory (sets download location). "
                             "Defaults to ~/.cache/huggingface. Set to scratch storage on cluster.")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Number of training images to export (default: all 4169).")
    parser.add_argument("--n_test", type=int, default=None,
                        help="Number of test images to export (default: all 439).")
    parser.add_argument("--image_ids", type=int, nargs="+", default=None,
                        help="Export only these image IDs (from both splits). Overrides --n_train/--n_test.")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)

    print("Loading restor/tcd from HuggingFace...")
    ds = load_dataset("restor/tcd", cache_dir=args.hf_cache_dir)

    train_json, train_img_dir = export_split(ds["train"], out_dir, args.n_train, "train", image_ids=args.image_ids)
    test_json, test_img_dir = export_split(ds["test"], out_dir, args.n_test, "test", image_ids=args.image_ids)

    print()
    print("Done. Paths match defaults in src/deepforest/conf/oam.yaml:")
    print(f"  train.csv_file:      {train_json}")
    print(f"  train.root_dir:      {train_img_dir}")
    print(f"  validation.csv_file: {test_json}")
    print(f"  validation.root_dir: {test_img_dir}")


if __name__ == "__main__":
    main()
