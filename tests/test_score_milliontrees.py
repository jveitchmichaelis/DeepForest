"""Per-source scoring of saved MillionTrees predictions."""

import pandas as pd
import pytest

from deepforest.scripts import score_milliontrees as score


def boxes(image_path, count, offset=0):
    """count non-overlapping 10 x 10 boxes in one image."""
    return pd.DataFrame({
        "image_path": image_path,
        "xmin": [20 * i + offset for i in range(count)],
        "ymin": 0, "xmax": [20 * i + 10 + offset for i in range(count)], "ymax": 10,
        "label": "Tree",
    })


def test_complete_images_reads_all_flag_values(tmp_path):
    official = tmp_path / "within-distribution.csv"
    pd.DataFrame({
        "filename": ["a.png", "b.png", "b.png", "c.png", "d.png"],
        "complete": ["True", "Image is completely annotated", "False", "False", "true"],
    }).to_csv(official, index=False)

    assert score.complete_images(official) == {"a.png", "b.png", "d.png"}


def test_score_by_source():
    # Source A is found perfectly, source B is missed
    truth = pd.concat([boxes("A/a1.png", 3), boxes("A/a2.png", 2), boxes("B/b1.png", 4)])
    predictions = pd.concat([boxes("A/a1.png", 3), boxes("A/a2.png", 2)]).assign(score=0.9)

    scores = score.score_by_source(truth, predictions, complete={"a1.png"}).set_index("group")

    assert scores.loc["A", ["images", "map_50", "box_recall"]].tolist() == [2, 1.0, 1.0]
    assert scores.loc["A", "complete_share"] == 0.5
    assert scores.loc["B", ["map_50", "box_recall"]].tolist() == [0.0, 0.0]
    assert scores.loc["macro average", "map_50"] == pytest.approx(0.5)
    assert scores.loc["all images", "images"] == 3
    assert scores.loc["all images", "map_50"] == pytest.approx(5 / 9, abs=0.02)
    assert scores.loc["complete images", ["images", "map_50"]].tolist() == [1, 1.0]


def test_images_without_folder_form_one_group():
    truth = pd.concat([boxes("x.tif", 2), boxes("y.tif", 2)])
    predictions = truth.assign(score=0.9)

    scores = score.score_by_source(truth, predictions)

    assert scores.group.tolist() == ["", "macro average", "all images"]
    assert "complete_share" not in scores


def test_workers_give_the_same_scores():
    truth = pd.concat([boxes("A/a1.png", 3), boxes("A/a2.png", 2), boxes("B/b1.png", 4)])
    predictions = pd.concat([boxes("A/a1.png", 3, offset=2), boxes("B/b1.png", 2)]).assign(score=0.9)

    serial = score.score_by_source(truth, predictions, workers=1)
    parallel = score.score_by_source(truth, predictions, workers=2)

    pd.testing.assert_frame_equal(serial, parallel)


def test_maskaware_precision_ignores_predictions_on_unlabelled_trees(tmp_path):
    """An unmatched prediction on tree pixels is not a false positive; one
    outside the tree coverage mask is."""
    pytest.importorskip("milliontrees")
    import numpy as np
    from PIL import Image

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # an unlabelled tree
    Image.fromarray(mask).save(tmp_path / "a1.png")
    truth = boxes("A/a1.png", 1)
    predictions = pd.concat([
        boxes("A/a1.png", 1),  # matches the labelled tree
        pd.DataFrame({"image_path": "A/a1.png", "xmin": [40, 80], "ymin": [40, 80],
                      "xmax": [60, 95], "ymax": [60, 95], "label": "Tree"}),
    ]).assign(score=0.9)

    scores = score.score_by_source(truth, predictions, masks_dir=tmp_path).set_index("group")

    assert scores.loc["A", "box_precision"] == pytest.approx(1 / 3)
    assert scores.loc["A", "maskaware_precision"] == pytest.approx(1 / 2)


def test_missing_mask_fails(tmp_path):
    truth = boxes("A/a1.png", 1)
    with pytest.raises(FileNotFoundError, match="masks missing"):
        score.score_by_source(truth, truth.assign(score=0.9), masks_dir=tmp_path)
