# Test Mask R-CNN polygon (instance segmentation) workflow
import os

import numpy as np
import pytest
import shapely
import torch

from deepforest import get_data, main, utilities
from deepforest.models import maskrcnn


@pytest.fixture()
def polygon_annotation_file():
    return get_data("coco_sample_file.json")


@pytest.fixture()
def polygon_root_dir():
    return os.path.dirname(get_data("coco_sample_file.json"))


# These tests use score_thresh=0.0 so a random-init model always returns
# something to assert on. That disables score filtering, so the model emits
# the full detections_per_img quota of full-resolution float masks — at the
# 2048x2048 test tiles, 300 detections is 5 GB for a single image. The
# detections of an untrained model are meaningless either way, so cap them.
TEST_DETECTIONS_PER_IMG = 10


def build_model(score_thresh=0.5, **kwargs):
    """Random-init Mask R-CNN (no weight download)."""
    kwargs.setdefault("box_detections_per_img", TEST_DETECTIONS_PER_IMG)

    return maskrcnn.MaskRCNN(
        backbone_weights=None,
        num_classes=1,
        label_dict={"tree": 0},
        score_thresh=score_thresh,
        **kwargs,
    )


def test_task_is_polygon():
    assert maskrcnn.MaskRCNN.task == "polygon"


def test_native_resolution_transform_does_not_resize():
    """Resize is disabled so the augmentation pipeline owns ground sample
    distance. min_size / max_size are inert even when passed explicitly."""
    model = build_model(score_thresh=0.0, min_size=400, max_size=512)
    model.eval()
    # 300x400 is below min_size and would be upscaled to 800 by torchvision
    predictions = model([torch.rand(3, 300, 400)])

    assert predictions[0]["masks"].shape[-2:] == (300, 400)


def test_inference_output_polygons():
    model = build_model(score_thresh=0.0, inference_output="polygons")
    model.eval()
    predictions = model([torch.rand(3, 200, 200)])

    assert sorted(predictions[0].keys()) == ["boxes", "labels", "polygons", "scores"]
    assert len(predictions[0]["polygons"]) == len(predictions[0]["labels"])
    for polygon in predictions[0]["polygons"]:
        assert isinstance(polygon, shapely.geometry.Polygon)


def test_inference_output_rejects_unknown_value():
    with pytest.raises(ValueError, match="inference_output"):
        build_model(inference_output="wat")


def test_predict_outputs_masks():
    model = build_model(score_thresh=0.0)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 400, 300)]
    predictions = model(x)

    assert len(predictions) == 2
    assert sorted(predictions[0].keys()) == ["boxes", "labels", "masks", "scores"]
    n = len(predictions[0]["labels"])
    # Masks are (N, 1, H, W) float logits before binarisation
    assert predictions[0]["masks"].shape == (n, 1, 300, 400)
    # Labels are shifted back to the zero-indexed DeepForest convention
    if n > 0:
        assert predictions[0]["labels"].min() >= 0


def test_training_label_shift_does_not_mutate_targets():
    model = build_model()
    model.train()
    images = [torch.rand(3, 200, 200)]
    targets = [
        {
            "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "masks": torch.zeros((1, 200, 200), dtype=torch.uint8),
        }
    ]
    targets[0]["masks"][0, 10:50, 10:50] = 1

    loss_dict = model(images, targets)

    assert "loss_mask" in loss_dict
    # The caller's target labels must remain zero-indexed
    assert targets[0]["labels"].tolist() == [0]


def test_config_fields_reach_torchvision():
    """Every maskrcnn.* config field maps onto a torchvision constructor
    argument, so overrides must show up on the built model."""
    m = main.deepforest(
        config_args={
            "architecture": "maskrcnn",
            "num_classes": 1,
            "label_dict": {"tree": 0},
            "detections_per_img": 123,
            "maskrcnn": {
                "rpn_post_nms_top_n_train": 1000,
                "rpn_pre_nms_top_n_test": 512,
                "rpn_post_nms_top_n_test": 512,
            },
        }
    )

    assert m.model.rpn._post_nms_top_n["training"] == 1000
    assert m.model.rpn._pre_nms_top_n["testing"] == 512
    assert m.model.rpn._post_nms_top_n["testing"] == 512
    assert m.model.roi_heads.detections_per_img == 123


def _dense_and_panoptic_targets():
    """The same two instances encoded both ways."""
    boxes = torch.tensor([[10, 10, 50, 50], [80, 80, 140, 140]], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.int64)

    dense = torch.zeros((2, 200, 200), dtype=torch.uint8)
    dense[0, 10:50, 10:50] = 1
    dense[1, 80:140, 80:140] = 1

    # PolygonDataset stores instance ids in one (H, W) map; ids are one-based
    # because 0 marks background. int32 matches what the dataset produces —
    # it casts off uint16 because torch comparisons don't support it on CPU.
    panoptic = torch.zeros((200, 200), dtype=torch.int32)
    panoptic[10:50, 10:50] = 1
    panoptic[80:140, 80:140] = 2

    dense_target = {"boxes": boxes, "labels": labels, "masks": dense}
    panoptic_target = {
        "boxes": boxes,
        "labels": labels,
        "panoptic_masks": panoptic,
        "unique_ids": torch.tensor([1, 2], dtype=torch.int32),
    }
    return dense_target, panoptic_target


def test_decode_panoptic_target_matches_dense_stack():
    dense_target, panoptic_target = _dense_and_panoptic_targets()

    decoded = utilities.decode_panoptic_target(panoptic_target)

    assert decoded.shape == dense_target["masks"].shape
    assert decoded.dtype == dense_target["masks"].dtype
    assert torch.equal(decoded, dense_target["masks"])


def test_decode_panoptic_target_empty():
    empty = {
        "panoptic_masks": torch.zeros((64, 64), dtype=torch.uint16),
        "unique_ids": torch.zeros((0,), dtype=torch.int64),
    }

    decoded = utilities.decode_panoptic_target(empty)

    assert decoded.shape == (0, 64, 64)


def test_panoptic_targets_give_same_losses_as_dense():
    """The panoptic encoding is a memory optimisation only. Training on it
    must produce exactly the losses the dense (N, H, W) stack would."""
    dense_target, panoptic_target = _dense_and_panoptic_targets()
    images = [torch.rand(3, 200, 200)]

    torch.manual_seed(0)
    model = build_model()
    model.train()

    torch.manual_seed(1)
    dense_losses = model(images, [dense_target])
    torch.manual_seed(1)
    panoptic_losses = model(images, [panoptic_target])

    assert dense_losses.keys() == panoptic_losses.keys()
    for key in dense_losses:
        assert torch.allclose(dense_losses[key], panoptic_losses[key]), key


def _make_polygon_model(tmp_path, polygon_annotation_file, polygon_root_dir):
    model = build_model(score_thresh=0.0)
    m = main.deepforest(
        model=model,
        config_args={
            "num_classes": 1,
            "label_dict": {"tree": 0},
            "detections_per_img": TEST_DETECTIONS_PER_IMG,
            # Mask R-CNN is broken on MPS with the torch version pinned here.
            "accelerator": "cpu",
            # The model never resizes, so without a crop this runs a forward
            # *and backward* over the full 2048x2048 sample tiles — tens of GB
            # on CPU. Real configs always crop (oam.yaml uses 1024); 256 keeps
            # the same code path at a size a test machine can hold.
            "train": {
                "augmentations": [{"RandomCrop": {"size": [256, 256], "p": 1.0}}]
            },
            "validation": {
                "augmentations": [{"CenterCrop": {"size": [256, 256], "p": 1.0}}]
            },
        },
    )
    m.set_labels({"tree": 0})
    m.config.train.csv_file = polygon_annotation_file
    m.config.train.root_dir = polygon_root_dir
    m.config.validation.csv_file = polygon_annotation_file
    m.config.validation.root_dir = polygon_root_dir
    m.config.train.fast_dev_run = True
    m.config.validation.val_accuracy_interval = 1
    m.config.log_root = str(tmp_path)
    m.create_trainer()
    return m


def test_polygon_train_and_validate(tmp_path, polygon_annotation_file, polygon_root_dir):
    m = _make_polygon_model(tmp_path, polygon_annotation_file, polygon_root_dir)
    assert m.model.task == "polygon"

    m.trainer.fit(m)
    m.trainer.validate(m)

    # Segmentation mAP and polygon recall/precision are logged
    logged = m.trainer.logged_metrics
    assert any("polygon" in key for key in logged) or "map" in logged


def test_polygon_predict_image_synthetic(polygon_root_dir):
    model = build_model(score_thresh=0.0)
    m = main.deepforest(
        model=model,
        config_args={
            "num_classes": 1,
            "label_dict": {"tree": 0},
            "detections_per_img": TEST_DETECTIONS_PER_IMG,
            # Mask R-CNN on MPS stalls badly here (96s at 6% CPU for a 150px
            # tile) and is broken outright on the torch version pinned in
            # this venv.
            "accelerator": "cpu",
        },
    )
    m.set_labels({"tree": 0})

    image = np.array(torch.randint(0, 255, (200, 200, 3), dtype=torch.uint8)).astype(
        "float32"
    )
    result = m.predict_image(image=image)

    # Random weights may or may not produce detections; if they do, they must
    # be polygons in the standard results schema.
    if result is not None:
        assert "geometry" in result.columns
        assert "label" in result.columns
        assert result.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")


def test_polygon_predict_tile():
    """predict_tile tiles the image and routes each window through the model,
    mosaicking polygon outputs across windows."""
    model = build_model(score_thresh=0.0)
    m = main.deepforest(
        model=model,
        config_args={
            "num_classes": 1,
            "label_dict": {"tree": 0},
            "detections_per_img": TEST_DETECTIONS_PER_IMG,
            # Mask R-CNN on MPS stalls badly here (96s at 6% CPU for a 150px
            # tile) and is broken outright on the torch version pinned in
            # this venv.
            "accelerator": "cpu",
        },
    )
    m.set_labels({"tree": 0})

    image = np.array(torch.randint(0, 255, (150, 150, 3), dtype=torch.uint8)).astype(
        "float32"
    )
    result = m.predict_tile(
        image=image, patch_size=100, patch_overlap=0.25, dataloader_strategy="single"
    )

    if result is not None:
        assert "geometry" in result.columns
        assert result.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
