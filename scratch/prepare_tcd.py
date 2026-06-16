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
from PIL import Image


def _verify_tif(path: str) -> bool:
    """Open with the same call training uses; return True iff it succeeds."""
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def _save_tif_atomic(pil_image, fpath: str, image_id: int) -> None:
    """Save a PIL image to fpath as TIFF, verify, then atomically rename.

    Raises RuntimeError if the saved file fails to roundtrip through PIL.
    """
    tmp_path = fpath + ".tmp"
    pil_image.convert("RGB").save(tmp_path, format="TIFF")
    if not _verify_tif(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Saved TIF for image_id={image_id} failed verification: {fpath}")
    os.replace(tmp_path, fpath)


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


def export_split(ds_split, out_dir: str, n: int | None, split_name: str, image_ids: list[int] | None = None, overwrite: bool = False):
    img_dir = os.path.join(out_dir, split_name, "images")
    os.makedirs(img_dir, exist_ok=True)

    images_meta = []
    all_annotations = []
    category_ids_seen: set[int] = set()

    json_path = os.path.join(out_dir, f"{split_name}.json")
    existing = None
    if image_ids is not None and os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        ann_id_offset = max((a["id"] for a in existing["annotations"]), default=0)
    else:
        ann_id_offset = 0

    if image_ids is not None:
        id_set = set(image_ids)
        ds_split = ds_split.filter(
            lambda x: int(x) in id_set,
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

        # Save image (atomic + verified). On re-runs without --image_ids, re-verify
        # the existing file and re-save if it's unreadable, so previously-corrupt
        # files self-heal instead of being skipped forever.
        fname = f"{image_id}.tif"
        fpath = os.path.join(img_dir, fname)
        needs_save = overwrite or image_ids is not None or not os.path.exists(fpath) or not _verify_tif(fpath)
        if needs_save:
            _save_tif_atomic(row["image"], fpath, image_id)

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
            category_ids_seen.add(int(ann["category_id"]))
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

    valid_category_ids = {c["id"] for c in CATEGORIES}
    unexpected = category_ids_seen - valid_category_ids
    assert not unexpected, (
        f"Unexpected category_ids in '{split_name}' annotations: {unexpected}. "
        f"CATEGORIES declares {valid_category_ids}."
    )

    if existing is not None:
        # Replace-by-image: drop any prior anns whose image we just re-exported,
        # then append the new ones. Same for images_meta. New ann IDs were
        # already offset above max(existing) so they can't collide.
        touched_image_ids = {img["id"] for img in images_meta}
        existing["images"] = [
            img for img in existing["images"] if img["id"] not in touched_image_ids
        ] + images_meta
        existing["annotations"] = [
            ann for ann in existing["annotations"] if ann["image_id"] not in touched_image_ids
        ] + all_annotations
        existing["categories"] = CATEGORIES
        coco_json = existing
    else:
        coco_json = {
            "images": images_meta,
            "annotations": all_annotations,
            "categories": CATEGORIES,
        }

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
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-save every TIF even if it already exists and verifies. "
                             "Use to force a clean rewrite after a corruption incident.")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)

    print("Loading restor/tcd from HuggingFace...")
    ds = load_dataset("restor/tcd", cache_dir=args.hf_cache_dir)

    train_json, train_img_dir = export_split(ds["train"], out_dir, args.n_train, "train", image_ids=args.image_ids, overwrite=args.overwrite)
    test_json, test_img_dir = export_split(ds["test"], out_dir, args.n_test, "test", image_ids=args.image_ids, overwrite=args.overwrite)

    print()
    print("Done. Paths match defaults in src/deepforest/conf/oam.yaml:")
    print(f"  train.csv_file:      {train_json}")
    print(f"  train.root_dir:      {train_img_dir}")
    print(f"  validation.csv_file: {test_json}")
    print(f"  validation.root_dir: {test_img_dir}")


if __name__ == "__main__":
    main()
