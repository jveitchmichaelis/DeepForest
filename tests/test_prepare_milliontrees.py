"""Contracts for the MillionTrees CSV adapter."""

import io
import urllib.request
import zipfile

import pandas as pd
import pytest
import truststore
from PIL import Image

from deepforest.scripts import prepare_milliontrees as mt
from deepforest.scripts.prepare_milliontrees import (
    convert_annotations,
    plan_image_layout,
    split_training_selection,
)


def annotations():
    """Small annotation table matching the official release schema."""
    return pd.DataFrame({
        "filename": ["a.png", "b.png", "c.png", "d.png", "e.tif", "f.tif"],
        "source": ["Example", "Example", "NeonTreeEvaluation", "NEON_unsupervised",
                   "NEON_benchmark", "NEON MultiTemporal"],
        "split": ["train", "validation", "train", "train", "test", "train"],
        "xmin": [0, 1, 2, 3, 4, 5], "ymin": [0, 1, 2, 3, 4, 5],
        "xmax": [10, 11, 12, 13, 14, 15], "ymax": [10, 11, 12, 13, 14, 15],
    })


def test_preserves_splits_and_excludes_neon():
    frames, audit = convert_annotations(annotations())

    assert frames["train"].image_path.tolist() == ["a.png"]
    assert frames["validation"].image_path.tolist() == ["b.png"]
    assert frames["train"].columns.tolist() == [
        "image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    assert frames["train"].label.tolist() == ["Tree"]
    assert frames["test"].empty
    assert audit["excluded_rows"] == 4


def test_uses_official_image_split_for_conflicting_annotation_rows():
    frame = annotations()
    frame.loc[1, "filename"] = "a.png"

    frames, audit = convert_annotations(frame)

    assert frames["train"].image_path.tolist() == ["a.png", "a.png"]
    assert frames["validation"].empty
    assert audit["conflicting_split_images"] == 1
    assert audit["reassigned_split_rows"] == 1


def test_rejects_invalid_boxes():
    frame = annotations()
    frame.loc[0, "xmax"] = -1

    with pytest.raises(ValueError, match="Invalid boxes"):
        convert_annotations(frame)


def test_layout_groups_sources_and_shards_large_sources():
    images = pd.DataFrame({
        "filename": ["a.png", "b.png", "c.png", "d.png"],
        "source": ["Source One", "Source One", "Source One", "Other"],
    })

    paths = plan_image_layout(images, max_per_directory=2)

    assert paths["d.png"] == "Other/d.png"
    assert all(paths[name].startswith("Source_One/") for name in ("a.png", "b.png", "c.png"))
    assert all(len(paths[name].split("/")) == 3 for name in ("a.png", "b.png", "c.png"))
    assert len({paths[name].split("/")[1] for name in ("a.png", "b.png", "c.png")}) >= 2


def test_selection_comes_only_from_official_train():
    train = pd.DataFrame({
        "image_path": ["a.png", "a.png", "b.png", "c.png"],
        "xmin": [0, 1, 0, 0],
    })
    sources = pd.DataFrame({
        "filename": ["a.png", "b.png", "c.png"],
        "source": ["A", "A", "B"],
    })

    remaining, selection = split_training_selection(train, sources)

    assert set(remaining.image_path).isdisjoint(selection.image_path)
    assert set(remaining.image_path) | set(selection.image_path) == set(train.image_path)
    assert selection.image_path.nunique() == 1
    assert "c.png" in set(remaining.image_path)


def test_export_resumes_after_interruption(tmp_path, monkeypatch):
    data_dir = tmp_path / "official"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True)

    frame = pd.DataFrame({
        "filename": ["a.png", "b.png", "c.png"],
        "source": ["Example"] * 3,
        "split": ["train", "train", "test"],
        "xmin": [0] * 3,
        "ymin": [0] * 3,
        "xmax": [2] * 3,
        "ymax": [2] * 3,
    })
    frame.to_csv(data_dir / "within-distribution.csv", index=False)
    for index, name in enumerate(frame.filename):
        Image.new("RGB", (4, 4), color=(index, 0, 0)).save(image_dir / name)

    original_sha256 = mt.sha256
    hashed_images = []

    def interrupted_sha256(path):
        if path.suffix == ".png":
            hashed_images.append(path.name)
            if len(hashed_images) == 2:
                raise KeyboardInterrupt

        return original_sha256(path)

    monkeypatch.setattr(mt, "sha256", interrupted_sha256)
    with pytest.raises(KeyboardInterrupt):
        mt.export_dataset(data_dir, tmp_path / "output", "within-distribution")

    hashed_images.clear()

    def resumed_sha256(path):
        if path.suffix == ".png":
            hashed_images.append(path.name)

        return original_sha256(path)

    monkeypatch.setattr(mt, "sha256", resumed_sha256)
    mt.export_dataset(data_dir, tmp_path / "output", "within-distribution")

    assert len(hashed_images) == 2


def test_organize_release_keeps_official_rows_and_split_overlap(tmp_path):
    data_dir = tmp_path / "official"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True)

    official = pd.DataFrame({
        "filename": ["a.png", "a.png", "b.png"],
        "source": ["Example", "Example", "Other"],
        "split": ["train", "test", "train"],
    })
    official.to_csv(data_dir / "within-distribution.csv", index=False)
    for name in ("a.png", "b.png", "unlisted.png"):
        Image.new("RGB", (4, 4)).save(image_dir / name)

    manifest = mt.organize_official_release(
        data_dir, tmp_path / "organized", "within-distribution"
    )
    mapping = pd.read_csv(tmp_path / "organized" / "image_map.csv")

    assert manifest["image_count"] == 3
    assert (tmp_path / "organized" / "within-distribution.csv").read_bytes() == (
        data_dir / "within-distribution.csv"
    ).read_bytes()
    assert set(mapping.organized_path) == {
        "Example/a.png", "Other/b.png", "Unmapped/unlisted.png"
    }
    assert all((tmp_path / "organized" / "images" / path).is_file()
               for path in mapping.organized_path)


def test_missing_data_dir_requests_matching_official_download(tmp_path):
    requested = tmp_path / "download" / "TreeBoxes_supervised_v0.23"

    release = mt.resolve_release_request(
        tmp_path, requested, version=None, size=None, include_unsupervised=False
    )

    assert release["download"] is True
    assert release["version"] == "0.23"
    assert release["root_dir"] == requested.parent
    assert release["data_dir"] == requested


def test_missing_data_dir_rejects_mismatched_version(tmp_path):
    requested = tmp_path / "download" / "TreeBoxes_supervised_v0.23"

    with pytest.raises(ValueError, match="does not match"):
        mt.resolve_release_request(
            tmp_path, requested, version="0.25", size=None, include_unsupervised=False
        )


def test_empty_release_directory_retries_official_download(tmp_path):
    requested = tmp_path / "download" / "TreeBoxes_supervised_v0.23"
    requested.mkdir(parents=True)

    release = mt.resolve_release_request(
        tmp_path, requested, version="0.23", size=None, include_unsupervised=False
    )

    assert release["download"] is True


def test_official_download_never_downgrades_to_http(monkeypatch):
    calls = []
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: None)
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, *args, **kwargs: calls.append(url))

    with mt.secure_official_downloads():
        urllib.request.urlretrieve("https://example.com/archive.zip")
        with pytest.raises(ValueError, match="HTTPS"):
            urllib.request.urlretrieve("http://example.com/archive.zip")

    assert calls == ["https://example.com/archive.zip"]


def test_resumes_partial_archive_with_http_range(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zipped:
        zipped.writestr("within-distribution.csv", "filename,source\na.png,Example\n")
    content = payload.getvalue()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(content[:10])
    requests = []

    class Response(io.BytesIO):
        def __init__(self, body, status, headers):
            super().__init__(body)
            self.status = status
            self.headers = headers

        def geturl(self):
            return "https://example.com/archive.zip"

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.get_method() == "HEAD":
            return Response(b"", 200, {"Content-Length": str(len(content))})

        return Response(
            content[10:],
            206,
            {"Content-Range": f"bytes 10-{len(content) - 1}/{len(content)}"},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    mt.resume_https_archive("https://example.com/archive.zip", archive)

    assert archive.read_bytes() == content
    assert requests[1].get_header("Range") == "bytes=10-"


def test_extracts_local_release_archive_and_reuses_completed_files(tmp_path, monkeypatch):
    archive = tmp_path / "TreeBoxes_supervised_v0.23.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("within-distribution.csv", "filename,source\na.png,Example\n")
        zipped.writestr("images/a.png", b"image bytes")

    extracted = tmp_path / "TreeBoxes_supervised_v0.23"
    mt.extract_release_archive(archive, extracted)

    assert (extracted / "images" / "a.png").read_bytes() == b"image bytes"
    assert (extracted / "within-distribution.csv").is_file()

    def unexpected_open(*args, **kwargs):
        raise AssertionError("Completed ZIP members should be reused")

    monkeypatch.setattr(zipfile.ZipFile, "open", unexpected_open)
    mt.extract_release_archive(archive, extracted)


def test_clips_intersecting_boxes_and_drops_fully_outside_boxes():
    boxes = pd.DataFrame({
        "xmin": [10.0, 390.0, 490.0],
        "ymin": [10.0, 20.0, 30.0],
        "xmax": [30.0, 420.0, 580.0],
        "ymax": [30.0, 40.0, 50.0],
    })

    clipped, valid = mt.clip_boxes_to_image(boxes, width=400, height=400)

    assert valid.tolist() == [True, True, False]
    assert clipped.loc[1, "xmax"] == 400


def test_duplicate_content_keeps_held_out_image_and_excludes_training_copy():
    frames = {
        "train": pd.DataFrame({"image_path": ["train.png", "unique.png"]}),
        "selection": pd.DataFrame({"image_path": ["selection.png"]}),
        "validation": pd.DataFrame({"image_path": []}),
        "test": pd.DataFrame({"image_path": ["test.png"]}),
    }
    images = pd.DataFrame({
        "official_filename": ["train.png", "unique.png", "selection.png", "test.png"],
        "split": ["train", "train", "selection", "test"],
        "sha256": ["same", "unique", "same", "same"],
    })

    retained, exclusions = mt.exclude_cross_split_duplicates(frames, images)

    assert frames["train"].image_path.tolist() == ["unique.png"]
    assert frames["selection"].empty
    assert frames["test"].image_path.tolist() == ["test.png"]
    assert retained.official_filename.tolist() == ["unique.png", "test.png"]
    assert set(exclusions.official_filename) == {"train.png", "selection.png"}



def test_neon_locations_reads_plots_and_tiles():
    names = pd.Series([
        "OSBS_029.tif",
        "Weecology_University_Florida/ABBY_069_2018_317158098_0_Weecology_University_Florida.png",
        "2018_TEAK_3_315000_4096000_image_12.tif",
        "Other/scene_01.png",
    ])

    assert mt.neon_locations(names).tolist()[:3] == ["OSBS_029", "ABBY_069", "TEAK_315000_4096000"]
    assert pd.isna(mt.neon_locations(names)[3])


def test_exclude_holdout_overlap_removes_same_plots_and_tiles(tmp_path):
    holdout = tmp_path / "test_cleaned.csv"
    pd.DataFrame({"image_path": ["OSBS_029.tif", "2018_TEAK_3_315000_4096000_image_12.tif"]}).to_csv(
        holdout, index=False)
    frames = {
        "train": pd.DataFrame({"image_path": [
            "OSBS_029_2019_Weecology.png", "OSBS_030_2019_Weecology.png", "other.png"]}),
        "test": pd.DataFrame({"image_path": [
            "2019_TEAK_3_315000_4096000_image_40_Weecology.png", "2019_TEAK_3_316000_4096000_image_1.png"]}),
    }

    frames, excluded, audit = mt.exclude_holdout_overlap(frames, holdout)

    assert frames["train"].image_path.tolist() == ["OSBS_030_2019_Weecology.png", "other.png"]
    assert frames["test"].image_path.tolist() == ["2019_TEAK_3_316000_4096000_image_1.png"]
    assert excluded.location.tolist() == ["TEAK_315000_4096000", "OSBS_029"]
    assert audit["excluded_holdout_images"] == 2


def test_exclude_holdout_overlap_needs_neon_names(tmp_path):
    holdout = tmp_path / "test.csv"
    pd.DataFrame({"image_path": ["scene_01.png"]}).to_csv(holdout, index=False)

    with pytest.raises(ValueError, match="No NEON plot or tile names"):
        mt.exclude_holdout_overlap({"train": pd.DataFrame({"image_path": ["a.png"]})}, holdout)
