"""Filter a DeepForest annotation CSV to only rows whose image is within a
maximum dimension.

Usage:
    uv run python src/scripts/filter_csv_by_image_size.py \
        --csv /path/to/annotations.csv \
        --root-dir /path/to/images \
        --max-size 1024 \
        --output /path/to/filtered.csv

Images where max(width, height) > --max-size are excluded entirely.
A summary of kept/dropped images is printed to stdout.
"""

import argparse
import os
import sys

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter annotation CSV to images within a maximum dimension."
    )
    parser.add_argument("--csv", required=True, help="Path to input annotation CSV.")
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Root directory for images. Defaults to the directory containing --csv.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        required=True,
        help="Maximum allowed dimension (longest side in pixels). Images exceeding this are dropped.",
    )
    parser.add_argument("--output", required=True, help="Path to write filtered CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)

    if "image_path" not in df.columns:
        sys.exit(f"CSV must have an 'image_path' column. Found: {list(df.columns)}")

    root_dir = args.root_dir or os.path.dirname(os.path.abspath(args.csv))

    unique_images = df["image_path"].unique()
    keep = []
    drop = []

    for img_name in unique_images:
        path = os.path.join(root_dir, img_name)
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as e:
            print(f"WARNING: could not open {path}: {e} — dropping")
            drop.append(img_name)
            continue

        if max(w, h) <= args.max_size:
            keep.append(img_name)
        else:
            drop.append(img_name)

    filtered = df[df["image_path"].isin(keep)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    filtered.to_csv(args.output, index=False)

    print(
        f"Kept {len(keep)}/{len(unique_images)} images "
        f"({len(drop)} dropped, max_size={args.max_size}px)."
    )
    print(f"Rows: {len(df)} -> {len(filtered)}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
