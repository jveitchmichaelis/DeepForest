"""Download official MillionTrees TreeBoxes and export DeepForest
annotations."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from PIL import Image

OFFICIAL_REVISION = "b6ebe7ecb53e56be7b036566b9e101aca49176f5"
BOXES = ["xmin", "ymin", "xmax", "ymax"]
MAX_IMAGES_PER_DIRECTORY = 50_000
# NEONTreeEvaluation is the NEON test set. The v0.23 release names it
# NEON_benchmark. NEON MultiTemporal repeats some of its plots in other years.
# Matched against lowercase source names with punctuation removed.
EXCLUDED_SOURCES = "neontreeeval|neonbenchmark|neonmultitemporal|unsupervised"


def convert_annotations(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Preserve official splits, excluding NEON test plots and weak labels.

    Coordinates are native-image Pascal VOC pixels; no resize is
    performed. Invalid annotations fail explicitly instead of removing
    trees from images.
    """
    required = {"filename", "source", "split", *BOXES}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing official columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing annotation values")

    sources = frame.source.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    excluded = sources.str.contains(EXCLUDED_SOURCES, regex=True)
    selected = frame.loc[~excluded].copy()

    if not set(selected.split).issubset({"train", "validation", "test"}):
        raise ValueError("Unknown official split names")

    image_splits = selected.drop_duplicates("filename").set_index("filename").split
    assigned_splits = selected.filename.map(image_splits)
    reassigned = selected.split.ne(assigned_splits)
    conflicting_names = sorted(selected.loc[reassigned, "filename"].unique())
    conflict_examples = [
        {
            "filename": name,
            "row_splits": sorted(
                selected.loc[selected.filename.eq(name), "split"].unique()
            ),
            "image_split": image_splits[name],
        }
        for name in conflicting_names[:20]
    ]
    selected["split"] = assigned_splits

    selected[BOXES] = selected[BOXES].apply(pd.to_numeric, errors="raise")
    valid = (
        np.isfinite(selected[BOXES]).all(axis=1)
        & selected.xmax.gt(selected.xmin)
        & selected.ymax.gt(selected.ymin)
    )
    if not valid.all():
        raise ValueError(f"Invalid boxes: {int((~valid).sum())}")

    selected = selected[["filename", "source", "split", *BOXES]].rename(
        columns={"filename": "image_path"}
    )
    selected["label"] = "Tree"
    frames = {
        split: selected.loc[selected.split.eq(split), ["image_path", *BOXES, "label"]]
        .sort_values(["image_path", *BOXES])
        .reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    if frames["train"].empty:
        raise ValueError("No retained train annotations")

    audit = {
        "excluded_rows": int(excluded.sum()),
        "excluded_sources": sorted(frame.loc[excluded, "source"].unique().tolist()),
        "retained_sources": sorted(selected.source.unique().tolist()),
        "conflicting_split_images": len(conflicting_names),
        "reassigned_split_rows": int(reassigned.sum()),
        "split_conflict_examples": conflict_examples,
        "source_split_rows": selected.groupby(["source", "split"])
        .size()
        .rename("rows")
        .reset_index()
        .to_dict("records"),
    }

    return frames, audit


def sha256(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clip_boxes_to_image(
    boxes: pd.DataFrame, width: int, height: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Clip image-intersecting boxes and flag boxes with zero clipped area."""
    clipped = boxes[BOXES].copy()
    clipped[["xmin", "xmax"]] = clipped[["xmin", "xmax"]].clip(0, width)
    clipped[["ymin", "ymax"]] = clipped[["ymin", "ymax"]].clip(0, height)
    valid = clipped.xmax.gt(clipped.xmin) & clipped.ymax.gt(clipped.ymin)

    return clipped, valid


def exclude_cross_split_duplicates(
    frames: dict[str, pd.DataFrame], images: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep identical content in the highest-priority held-out split only."""
    priority = {"train": 0, "selection": 1, "validation": 2, "test": 3}
    ranked = images.assign(priority=images.split.map(priority))
    highest = ranked.groupby("sha256").priority.transform("max")
    excluded = ranked.loc[ranked.priority.lt(highest)].copy()
    excluded["retained_split"] = highest.loc[excluded.index].map(
        {value: name for name, value in priority.items()}
    )
    excluded = excluded.drop(columns="priority").sort_values(
        ["split", "official_filename"]
    )
    retained = ranked.loc[ranked.priority.eq(highest)].drop(columns="priority")

    for split, frame in frames.items():
        names = excluded.loc[excluded.split.eq(split), "official_filename"]
        frames[split] = frame.loc[~frame.image_path.isin(names)].reset_index(drop=True)

    return retained.reset_index(drop=True), excluded.reset_index(drop=True)


def plan_image_layout(
    images: pd.DataFrame, max_per_directory: int = MAX_IMAGES_PER_DIRECTORY
) -> dict[str, str]:
    """Map official flat filenames to source folders, sharding large
    sources."""
    if max_per_directory < 1:
        raise ValueError("max_per_directory must be positive")

    pairs = images[["filename", "source"]].drop_duplicates()
    if pairs.filename.duplicated().any():
        raise ValueError("An image filename belongs to multiple sources")

    source_names = sorted(pairs.source.unique())
    slugs = {
        source: re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("._-") or "source"
        for source in source_names
    }
    slug_counts = Counter(slug.lower() for slug in slugs.values())
    for source in source_names:
        if slug_counts[slugs[source].lower()] > 1:
            suffix = hashlib.sha256(source.encode()).hexdigest()[:8]
            slugs[source] = f"{slugs[source]}_{suffix}"

    layout = {}
    for source, group in pairs.groupby("source", sort=True):
        names = sorted(group.filename)
        if any(Path(name).name != name or name in {".", ".."} for name in names):
            raise ValueError(f"Expected flat official image filenames for {source}")

        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        depth = 0
        if len(names) > max_per_directory:
            depth = 2
            while (
                max(Counter(digest[:depth] for digest in hashes.values()).values())
                > max_per_directory
            ):
                depth += 2
                if depth > 64:
                    raise ValueError(
                        f"Cannot shard source {source} below directory limit"
                    )

        for name in names:
            parent = Path(slugs[source])
            if depth:
                parent /= hashes[name][:depth]
            layout[name] = (parent / name).as_posix()

    return layout


def organize_images(image_root: Path, output_root: Path, layout: dict[str, str]) -> None:
    """Build a reusable DeepForest tree with hard links where possible."""
    for filename, relative in sorted(layout.items()):
        source = image_root / filename
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            if os.path.samefile(source, destination) or sha256(source) == sha256(
                destination
            ):
                continue
            raise ValueError(f"Organized image differs from source: {destination}")

        try:
            os.link(source, destination)
        except OSError:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
                temporary = Path(tmp.name)

            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def organize_official_release(
    data_dir: Path,
    output: Path,
    split_scheme: str,
    progress_interval_seconds: float = 30,
) -> dict:
    """Group every shipped image by source without changing official CSV
    rows."""
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    source_csv = data_dir / f"{split_scheme}.csv"
    print(f"Reading {source_csv}", flush=True)
    annotations = pd.read_csv(
        source_csv, usecols=["filename", "source"], low_memory=False
    )
    sources = (
        annotations.dropna()
        .groupby("filename")
        .source.agg(lambda values: sorted(set(values)))
    )

    image_root = data_dir / "images"
    image_files = sorted(path for path in image_root.iterdir() if path.is_file())
    if any(path.is_dir() for path in image_root.iterdir()):
        raise ValueError(f"Expected flat official images directory: {image_root}")

    image_names = {path.name for path in image_files}
    missing = set(sources.index) - image_names
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CSV images missing from {image_root}; first: {sorted(missing)[0]}"
        )

    assigned = []
    for path in image_files:
        candidates = sources.get(path.name, [])
        source = (
            candidates[0]
            if len(candidates) == 1
            else ("Unmapped" if not candidates else "Shared")
        )
        assigned.append({"filename": path.name, "source": source})

    mapping = pd.DataFrame(assigned)
    layout = plan_image_layout(mapping)
    mapping["organized_path"] = mapping.filename.map(layout)

    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name == "archive.zip":
            continue

        destination = output / path.name
        if not destination.exists():
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)

    last_report = time.monotonic()
    print(f"Organizing {len(image_files)} official images", flush=True)
    for index, path in enumerate(image_files, start=1):
        organize_images(image_root, output / "images", {path.name: layout[path.name]})

        now = time.monotonic()
        if index % 100 == 0 or now - last_report >= progress_interval_seconds:
            print(f"Organized {index}/{len(image_files)} images", flush=True)
            last_report = now

    mapping.to_csv(output / "image_map.csv", index=False)
    manifest = {
        "mode": "organize-only",
        "official_data_dir": str(data_dir.resolve()),
        "split_scheme_for_source_names": split_scheme,
        "image_count": len(image_files),
        "unmapped_images": int(mapping.source.eq("Unmapped").sum()),
        "shared_source_images": int(mapping.source.eq("Shared").sum()),
        "image_root": str((output / "images").resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Organized {len(image_files)} images in {output}", flush=True)

    return manifest


def split_training_selection(
    train: pd.DataFrame, source_by_image: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve a deterministic per-source sample from official training
    images."""
    sources = source_by_image.drop_duplicates().set_index("filename").source
    if sources.index.has_duplicates:
        raise ValueError("An image filename belongs to multiple sources")

    candidates = pd.DataFrame({"filename": train.image_path.unique()})
    candidates["source"] = candidates.filename.map(sources)
    held_out = set()
    for _, group in candidates.groupby("source", sort=True):
        if len(group) < 2:
            continue

        ranked = sorted(
            group.filename,
            key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name),
        )
        count = min(len(ranked) - 1, max(1, round(len(ranked) * 0.1)))
        held_out.update(ranked[:count])

    selected = train.image_path.isin(held_out)
    return train.loc[~selected].copy(), train.loc[selected].copy()


def export_dataset(
    data_dir: Path,
    output: Path,
    split_scheme: str,
    progress_interval_seconds: float = 30,
) -> dict:
    """Validate imagery and write deterministic CSVs plus a provenance
    manifest."""
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    source = data_dir / f"{split_scheme}.csv"
    print(f"Reading {source}", flush=True)
    official = pd.read_csv(source, low_memory=False)
    frames, audit = convert_annotations(official)

    retained = pd.concat(frames.values()).image_path.unique()
    source_keys = official.source.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    retained_metadata = official.loc[
        official.filename.isin(retained)
        & ~source_keys.str.contains(EXCLUDED_SOURCES, regex=True),
        ["filename", "source"],
    ].drop_duplicates("filename")
    layout = plan_image_layout(retained_metadata)
    frames["train"], frames["selection"] = split_training_selection(
        frames["train"], retained_metadata
    )
    if frames["selection"].empty:
        raise ValueError("No official training images available for model selection")

    image_root = (data_dir / "images").resolve()
    organized_root = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(source)
    fingerprint = json.dumps(
        {
            "source_csv_sha256": source_digest,
            "split_scheme": split_scheme,
            "image_root": str(image_root),
            "layout_version": 1,
        },
        sort_keys=True,
    )

    images = []
    clipped_boxes = 0
    dropped_outside_boxes = 0
    skipped_empty_images = 0
    invalid_rows = {split: [] for split in frames}
    total_images = sum(frame.image_path.nunique() for frame in frames.values())
    processed = 0
    reused = 0
    last_report = time.monotonic()
    progress_path = output / "progress.sqlite3"

    with closing(sqlite3.connect(progress_path)) as progress:
        progress.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        progress.execute(
            "CREATE TABLE IF NOT EXISTS images ("
            "filename TEXT PRIMARY KEY, split TEXT NOT NULL, relative_path TEXT NOT NULL, "
            "source_size INTEGER NOT NULL, source_mtime_ns INTEGER NOT NULL, "
            "sha256 TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL)"
        )
        recorded = progress.execute(
            "SELECT value FROM metadata WHERE key = 'fingerprint'"
        ).fetchone()
        if recorded and recorded[0] != fingerprint:
            raise ValueError(
                f"Source CSV or layout changed; use a fresh target or remove {progress_path}"
            )
        if not recorded:
            progress.execute(
                "INSERT INTO metadata (key, value) VALUES ('fingerprint', ?)",
                (fingerprint,),
            )
        progress.commit()

        cached = {
            row[0]: row[1:]
            for row in progress.execute(
                "SELECT filename, split, relative_path, source_size, "
                "source_mtime_ns, sha256, width, height FROM images"
            )
        }
        print(
            f"Preparing {total_images} images ({len(cached)} recorded); progress every 100 images or {progress_interval_seconds:g} seconds",
            flush=True,
        )

        for split, frame in frames.items():
            for filename, rows in frame.groupby("image_path", sort=True):
                path = (image_root / filename).resolve()
                if not path.is_relative_to(image_root):
                    raise ValueError(f"Image path escapes the dataset: {filename}")

                stat = path.stat()
                relative = layout[filename]
                destination = organized_root / relative
                prior = cached.get(filename)
                valid_cache = (
                    prior is not None
                    and prior[:4] == (split, relative, stat.st_size, stat.st_mtime_ns)
                    and destination.is_file()
                )
                if valid_cache:
                    valid_cache = (
                        os.path.samefile(path, destination)
                        or sha256(destination) == prior[4]
                    )

                if valid_cache:
                    digest, width, height = prior[4:]
                    reused += 1
                else:
                    with Image.open(path) as image:
                        width, height = image.size

                    digest = sha256(path)
                    organize_images(image_root, organized_root, {filename: relative})
                    progress.execute(
                        "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            filename,
                            split,
                            relative,
                            stat.st_size,
                            stat.st_mtime_ns,
                            digest,
                            width,
                            height,
                        ),
                    )
                    progress.commit()

                clipped, valid = clip_boxes_to_image(rows, width, height)
                dropped_outside_boxes += int((~valid).sum())
                invalid_rows[split].extend(rows.index[~valid].tolist())
                if valid.any():
                    clipped_boxes += int(
                        clipped.loc[valid].ne(rows.loc[valid, BOXES]).any(axis=1).sum()
                    )
                    frames[split].loc[rows.index[valid], BOXES] = clipped.loc[valid]
                    images.append(
                        {
                            "image_path": relative,
                            "official_filename": filename,
                            "split": split,
                            "sha256": digest,
                            "width": width,
                            "height": height,
                        }
                    )
                else:
                    skipped_empty_images += 1

                processed += 1
                now = time.monotonic()
                if processed % 100 == 0 or now - last_report >= progress_interval_seconds:
                    print(
                        f"Prepared {processed}/{total_images} images ({reused} reused)",
                        flush=True,
                    )
                    last_report = now

    print(f"Validated {processed}/{total_images} images ({reused} reused)", flush=True)

    for split, invalid in invalid_rows.items():
        frames[split] = frames[split].drop(index=invalid).reset_index(drop=True)
    if frames["train"].empty or frames["selection"].empty:
        raise ValueError("No usable train or selection annotations remain")

    image_manifest = pd.DataFrame(images)
    duplicate_content_groups = int(
        image_manifest.groupby("sha256").split.nunique().gt(1).sum()
    )
    original_box_count = sum(len(frame) for frame in frames.values())
    image_manifest, excluded_duplicates = exclude_cross_split_duplicates(
        frames, image_manifest
    )
    if frames["train"].empty or frames["selection"].empty:
        raise ValueError("No usable train or selection annotations remain")

    for split, frame in frames.items():
        frame.assign(image_path=frame.image_path.map(layout)).to_csv(
            output / f"{split}.csv", index=False
        )
    image_manifest.to_csv(output / "images.csv", index=False)
    excluded_duplicates.to_csv(output / "excluded_duplicates.csv", index=False)

    audit.update(
        {
            "official_repository": "https://github.com/weecology/MillionTrees",
            "expected_official_revision": OFFICIAL_REVISION,
            "split_scheme": split_scheme,
            "source_csv_sha256": source_digest,
            "image_root": str(organized_root),
            "official_image_root": str(image_root),
            "max_images_per_directory": MAX_IMAGES_PER_DIRECTORY,
            "clipped_boxes": clipped_boxes,
            "dropped_outside_boxes": dropped_outside_boxes,
            "skipped_empty_images": skipped_empty_images,
            "duplicate_content_groups": duplicate_content_groups,
            "excluded_duplicate_images": len(excluded_duplicates),
            "excluded_duplicate_boxes": original_box_count
            - sum(len(frame) for frame in frames.values()),
            "selection_method": "10 percent of official train images per source, ranked by SHA256 filename; sources with one image remain in train",
            "artifacts": {
                f"{name}.csv": sha256(output / f"{name}.csv")
                for name in (*frames, "images", "excluded_duplicates")
            },
            "counts": {
                split: {"boxes": len(frame), "images": frame.image_path.nunique()}
                for split, frame in frames.items()
            },
        }
    )

    return audit


def resolve_release_request(
    target: Path,
    data_dir: Path | None,
    version: str | None,
    size: str | None,
    include_unsupervised: bool,
) -> dict[str, Path | str | bool]:
    """Resolve an existing release or an official download destination."""
    requested = data_dir.expanduser().resolve() if data_dir else None
    match = (
        re.fullmatch(r"(Mini|Small)?TreeBoxes(_supervised)?_v(\d+\.\d+)", requested.name)
        if requested
        else None
    )

    if match:
        prefix, supervised, folder_version = match.groups()
        folder_size = {"Mini": "mini", "Small": "small", None: "full"}[prefix]
        if version is not None and version != folder_version:
            raise ValueError(
                f"--version {version} does not match --data-dir release {folder_version}"
            )
        if size is not None and size != folder_size:
            raise ValueError(
                f"--size {size} does not match --data-dir size {folder_size}"
            )
        if folder_size == "full" and bool(supervised) == include_unsupervised:
            raise ValueError(
                "--include-unsupervised does not match --data-dir archive type"
            )

        version = folder_version
        size = folder_size

    version = version or "0.25"
    size = size or "full"
    expected_name = (
        {"mini": "MiniTreeBoxes", "small": "SmallTreeBoxes", "full": "TreeBoxes"}[size]
        + ("_supervised" if size == "full" and not include_unsupervised else "")
        + f"_v{version}"
    )

    release_ready = (
        requested is not None
        and (requested / "within-distribution.csv").is_file()
        and (requested / "images").is_dir()
    )
    if requested and not release_ready and requested.name != expected_name:
        raise ValueError(
            f"Incomplete --data-dir must name the official release folder {expected_name}"
        )

    return {
        "data_dir": requested or (target / "download" / expected_name).resolve(),
        "root_dir": requested.parent if requested else (target / "download").resolve(),
        "version": version,
        "size": size,
        "download": not release_ready,
    }


@contextmanager
def secure_official_downloads():
    """Use system CAs and reject the upstream HTTP fallback during download."""
    import truststore

    truststore.inject_into_ssl()
    original = urllib.request.urlretrieve

    def https_only(url, *args, **kwargs):
        if urlsplit(str(url)).scheme.lower() != "https":
            raise ValueError(f"Refusing non-HTTPS official download: {url}")

        return original(url, *args, **kwargs)

    urllib.request.urlretrieve = https_only
    try:
        yield
    finally:
        urllib.request.urlretrieve = original


def resume_https_archive(
    url: str, archive: Path, progress_interval_seconds: float = 30
) -> None:
    """Download an official ZIP over HTTPS, resuming a partial file by
    range."""
    if urlsplit(url).scheme.lower() != "https":
        raise ValueError(f"Refusing non-HTTPS official download: {url}")

    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=60
    ) as response:
        if urlsplit(response.geturl()).scheme.lower() != "https":
            raise ValueError(f"Refusing non-HTTPS redirect: {response.geturl()}")
        total = int(response.headers["Content-Length"])

    archive.parent.mkdir(parents=True, exist_ok=True)
    existing = archive.stat().st_size if archive.exists() else 0
    if existing == total:
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"Downloaded archive is not a valid ZIP: {archive}")

        print(f"Using complete archive: {archive}", flush=True)
        return
    if existing > total:
        raise ValueError(f"Local archive exceeds official size: {archive}")

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {url} to {archive} from byte {existing}/{total}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        if urlsplit(response.geturl()).scheme.lower() != "https":
            raise ValueError(f"Refusing non-HTTPS redirect: {response.geturl()}")
        status = response.status
        if existing and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise ValueError(f"Unexpected HTTP range response: {content_range}")
            mode = "ab"
        elif status == 200:
            mode = "wb"
            existing = 0
        else:
            raise ValueError(f"Unexpected HTTP download status: {status}")

        downloaded = existing
        last_report = time.monotonic()
        with archive.open(mode) as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_report >= progress_interval_seconds:
                    print(f"Downloaded {downloaded}/{total} bytes", flush=True)
                    last_report = now

    if archive.stat().st_size != total:
        raise OSError(f"Download interrupted at {archive.stat().st_size}/{total} bytes")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Downloaded archive is not a valid ZIP: {archive}")

    print(f"Downloaded complete archive: {archive}", flush=True)


def official_archive_url(version: str, size: str, include_unsupervised: bool) -> str:
    """Read the release URL from the installed official TreeBoxes package."""
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

    versions = TreeBoxesDataset._versions_dict
    if version not in versions:
        raise ValueError(
            f"Version {version} is unavailable in installed MillionTrees; available: {sorted(versions)}"
        )

    info = versions[version]
    if size == "full" and not include_unsupervised:
        url = info.get("supervised_download_url")
    else:
        url = info.get("download_url")
        if size != "full" and url:
            prefix = {"mini": "Mini", "small": "Small"}[size]
            url = url.replace(
                f"TreeBoxes_v{version}.zip", f"{prefix}TreeBoxes_v{version}.zip"
            )

    if not url:
        raise ValueError(f"No official download URL for TreeBoxes v{version} ({size})")

    return url


def extract_release_archive(
    archive: Path, data_dir: Path, progress_interval_seconds: float = 30
) -> None:
    """Extract a local official ZIP with per-file resume checkpoints."""
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_stat = archive.stat()
    fingerprint = json.dumps(
        {
            "archive": str(archive.resolve()),
            "size": archive_stat.st_size,
            "mtime_ns": archive_stat.st_mtime_ns,
        },
        sort_keys=True,
    )
    progress_path = data_dir / "extract_progress.sqlite3"

    with (
        closing(sqlite3.connect(progress_path)) as progress,
        zipfile.ZipFile(archive) as zipped,
    ):
        progress.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        progress.execute(
            "CREATE TABLE IF NOT EXISTS completed ("
            "name TEXT PRIMARY KEY, size INTEGER NOT NULL, crc INTEGER NOT NULL)"
        )
        recorded = progress.execute(
            "SELECT value FROM metadata WHERE key = 'fingerprint'"
        ).fetchone()
        if recorded and recorded[0] != fingerprint:
            raise ValueError(
                f"Archive changed; remove stale extraction journal: {progress_path}"
            )
        if not recorded:
            progress.execute(
                "INSERT INTO metadata (key, value) VALUES ('fingerprint', ?)",
                (fingerprint,),
            )
            progress.commit()

        completed = {
            row[0]: (row[1], row[2])
            for row in progress.execute("SELECT name, size, crc FROM completed")
        }
        members = [member for member in zipped.infolist() if not member.is_dir()]
        last_report = time.monotonic()
        reused = 0
        print(f"Extracting {len(members)} archive files from {archive}", flush=True)

        for index, member in enumerate(members, start=1):
            destination = (data_dir / member.filename).resolve()
            if not destination.is_relative_to(data_dir.resolve()):
                raise ValueError(
                    f"Archive path escapes release directory: {member.filename}"
                )

            if (
                completed.get(member.filename) == (member.file_size, member.CRC)
                and destination.is_file()
                and destination.stat().st_size == member.file_size
            ):
                reused += 1
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, delete=False
                ) as temporary_stream:
                    temporary = Path(temporary_stream.name)

                    try:
                        with zipped.open(member) as source:
                            shutil.copyfileobj(source, temporary_stream)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise

                os.replace(temporary, destination)
                progress.execute(
                    "INSERT OR REPLACE INTO completed VALUES (?, ?, ?)",
                    (member.filename, member.file_size, member.CRC),
                )
                progress.commit()

            now = time.monotonic()
            if index % 100 == 0 or now - last_report >= progress_interval_seconds:
                print(
                    f"Extracted {index}/{len(members)} files ({reused} reused)",
                    flush=True,
                )
                last_report = now

    print(f"Extracted {len(members)} files ({reused} reused)", flush=True)


def main() -> None:
    """Download once, export annotations, and create a DeepForest CLI
    launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--version", help="Official release; inferred from --data-dir when possible."
    )
    parser.add_argument("--size", choices=["full", "small", "mini"])
    parser.add_argument(
        "--organize-only",
        action="store_true",
        help="Preserve official CSVs and organize all shipped images by source.",
    )
    parser.add_argument(
        "--include-unsupervised",
        action="store_true",
        help="Download the full archive, including unsupervised images, in organize-only mode.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30,
        metavar="SECONDS",
        help="Maximum interval between preparation progress messages.",
    )
    parser.add_argument(
        "--split-scheme",
        default="within-distribution",
        choices=["within-distribution", "out-of-distribution"],
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Official release directory; download it there if absent.",
    )
    args = parser.parse_args()
    if args.include_unsupervised and not args.organize_only:
        parser.error("--include-unsupervised requires --organize-only")

    target = args.target.expanduser().resolve()
    release = resolve_release_request(
        target, args.data_dir, args.version, args.size, args.include_unsupervised
    )
    provenance = {
        "requested_version": release["version"],
        "size": release["size"],
    }
    local_archive = release["data_dir"] / f"{release['data_dir'].name}.zip"
    extraction_journal = release["data_dir"] / "extract_progress.sqlite3"
    if local_archive.is_file() and (release["download"] or extraction_journal.is_file()):
        extract_release_archive(
            local_archive, release["data_dir"], args.progress_interval
        )
        data_dir = release["data_dir"]
        provenance["download"] = "existing official archive"
    elif release["download"]:
        from importlib.metadata import distribution

        from milliontrees import get_dataset

        package = distribution("milliontrees")
        provenance["milliontrees_package_version"] = package.version
        provenance["milliontrees_direct_url"] = package.read_text("direct_url.json")

        with secure_official_downloads():
            url = official_archive_url(
                release["version"], release["size"], args.include_unsupervised
            )
            resume_https_archive(
                url, release["data_dir"] / "archive.zip", args.progress_interval
            )
            dataset = get_dataset(
                "TreeBoxes",
                version=release["version"],
                root_dir=str(release["root_dir"]),
                download=True,
                split_scheme=args.split_scheme,
                mini=release["size"] == "mini",
                small=release["size"] == "small",
                include_unsupervised=args.include_unsupervised,
                exclude_sources=[]
                if args.organize_only
                else [
                    "*NeonTreeEval*",
                    "*NEON_benchmark*",
                    "*NEON MultiTemporal*",
                    "*unsupervised*",
                ],
            )
        data_dir = Path(dataset.data_dir)
        if data_dir.resolve() != release["data_dir"]:
            raise ValueError(
                f"Official loader downloaded to {data_dir}, expected {release['data_dir']}"
            )
        provenance["resolved_version"] = dataset.version
        provenance["download"] = "official loader"
    else:
        data_dir = release["data_dir"]
        provenance["download"] = "existing local release"

    if not (data_dir / f"{args.split_scheme}.csv").is_file():
        raise FileNotFoundError(
            f"Official release is incomplete: {data_dir / f'{args.split_scheme}.csv'}"
        )

    if args.organize_only:
        output = target / "organized"
        manifest = organize_official_release(
            data_dir, output, args.split_scheme, args.progress_interval
        )
        manifest.update(provenance)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return

    output = target / "deepforest"
    audit = export_dataset(data_dir, output, args.split_scheme, args.progress_interval)
    audit.update(provenance)
    (output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")

    overrides = [
        f"train.csv_file={json.dumps(str(output / 'train.csv'))}",
        f"validation.csv_file={json.dumps(str(output / 'selection.csv'))}",
        f"train.root_dir={json.dumps(audit['image_root'])}",
        f"validation.root_dir={json.dumps(audit['image_root'])}",
        f"log_root={json.dumps(str(target / 'runs'))}",
    ]
    command = shlex.join(
        [
            "uv",
            "run",
            "deepforest",
            "--config-name",
            "milliontrees-dinov3",
            "train",
            "--export-hf",
            *overrides,
        ]
    )

    (output / "train.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "# Run from the DeepForest checkout; pass model.name and other Hydra overrides.\n"
        'export PL_GLOBAL_SEED="${PL_GLOBAL_SEED:-0}"\n'
        f'exec {command} "$@"\n'
    )

    print(f"Prepared {output}; counts: {audit['counts']}")
    print(f"Train: bash {shlex.quote(str(output / 'train.sh'))} model.name=<checkpoint>")


if __name__ == "__main__":
    main()
