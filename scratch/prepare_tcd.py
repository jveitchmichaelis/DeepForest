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


CATEGORY = [{"id": 1, "name": "tree", "supercategory": "tree"}]


def export_split(ds_split, out_dir: str, n: int | None, split_name: str):
    img_dir = os.path.join(out_dir, split_name, "images")
    os.makedirs(img_dir, exist_ok=True)

    images_meta = []
    all_annotations = []
    ann_id_offset = 0

    n_total = len(ds_split) if n is None else min(n, len(ds_split))
    print(f"Exporting {n_total} images from '{split_name}' split...")

    for i in range(n_total):
        row = ds_split[i]
        image_id = int(row["image_id"])
        height = int(row["height"])
        width = int(row["width"])

        # Save image
        fname = f"{image_id}.tif"
        fpath = os.path.join(img_dir, fname)
        if not os.path.exists(fpath):
            row["image"].save(fpath)

        images_meta.append({"id": image_id, "file_name": fname, "height": height, "width": width})

        # Parse per-image annotations; skip RLE-encoded crowd annotations (not supported).
        raw_anns = json.loads(row["coco_annotations"])
        for ann in raw_anns:
            if not isinstance(ann.get("segmentation"), list):
                continue
            ann_id_offset += 1
            entry = {
                "id": ann_id_offset,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": ann["segmentation"],
                "area": ann.get("area", 0),
                "bbox": ann["bbox"],
                "iscrowd": 0,
            }
            all_annotations.append(entry)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_total}")

    coco_json = {
        "images": images_meta,
        "annotations": all_annotations,
        "categories": CATEGORY,
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
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)

    print("Loading restor/tcd from HuggingFace...")
    ds = load_dataset("restor/tcd", cache_dir=args.hf_cache_dir)

    train_json, train_img_dir = export_split(ds["train"], out_dir, args.n_train, "train")
    test_json, test_img_dir = export_split(ds["test"], out_dir, args.n_test, "test")

    print()
    print("Done. Paths match defaults in src/deepforest/conf/oam.yaml:")
    print(f"  train.csv_file:      {train_json}")
    print(f"  train.root_dir:      {train_img_dir}")
    print(f"  validation.csv_file: {test_json}")
    print(f"  validation.root_dir: {test_img_dir}")


if __name__ == "__main__":
    main()
