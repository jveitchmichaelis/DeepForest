# test retinanet
import copy
import os

import pytest
import torch

from deepforest.models import retinanet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Empty tester from https://github.com/datumbox/vision/blob/06ebee1a9f10c76d8ac5768fd578362dd5ace6e9/test/test_models_detection_negative_samples.py#L14
def _make_empty_sample():
    images = [torch.rand((3, 100, 100), dtype=torch.float32)]
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    negative_target = {
        "boxes": boxes,
        "labels": torch.zeros(0, dtype=torch.int64),
        "image_id": 4,
        "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        "iscrowd": torch.zeros((0,), dtype=torch.int64)
    }

    targets = [negative_target]
    return images, targets


def test_retinanet(config):
    r = retinanet.Model(config)
    assert r


def retinanet_check_model(config):
    r = retinanet.Model(config)
    r.check_model()


@pytest.mark.parametrize("num_classes", [1, 2, 10])
def test_create_model(config, num_classes):
    config.num_classes = num_classes
    config.label_dict = {f"{i}": i for i in range(num_classes)}
    retinanet_model = retinanet.Model(config).create_model()
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)


def test_forward_empty(config):
    r = retinanet.Model(config)
    model = r.create_model()
    image, targets = _make_empty_sample()
    loss = model(image, targets)
    assert torch.equal(loss["bbox_regression"], torch.tensor(0.))


# Can we update parameters after training
def test_maintain_parameters(config):
    config.score_thresh = 0.4
    retinanet_model = retinanet.Model(config).create_model()
    assert retinanet_model.score_thresh == config.score_thresh
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)
    assert retinanet_model.score_thresh == config.score_thresh

    retinanet_model.score_thresh = 0.9
    retinanet_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = retinanet_model(x)
    assert retinanet_model.score_thresh == 0.9

def test_retinanet_override_class(tmpdir):
    # Check we correctly override num_classes and label_dict
    model = retinanet.RetinaNetHub.from_pretrained("weecology/deepforest-tree", num_classes=2, label_dict={"Tree": 0, "Shrub": 1})
    assert model.label_dict == {"Tree": 0, "Shrub": 1}
    assert model.head.classification_head.num_classes == 2

    # Confirm it persists when reloading
    model.save_pretrained(tmpdir)
    model2 = retinanet.RetinaNetHub.from_pretrained(tmpdir)
    assert model2.label_dict == {"Tree": 0, "Shrub": 1}
    assert model2.head.classification_head.num_classes == 2

def test_retinanet_override_class_error():
    # Mismatch between label_dict and class count
    with pytest.raises(ValueError):
        retinanet.RetinaNetHub.from_pretrained("weecology/deepforest-tree", num_classes=1, label_dict={"Tree": 0, "Shrub": 1})


@pytest.mark.parametrize("resize", [True, False])
def test_create_model_resize(config, resize):
    """With model.resize False, the model keeps each image's size."""
    config = copy.deepcopy(config)
    config.num_classes = 1
    config.label_dict = {"Tree": 0}
    config.model.resize = resize
    model = retinanet.Model(config).create_model()
    model.eval()

    images, _ = model.transform([torch.rand(3, 400, 600)])

    assert images.image_sizes == [(800, 1200) if resize else (400, 600)]


@pytest.mark.parametrize("freeze_backbone", [False, True])
def test_freeze_backbone(config, monkeypatch, freeze_backbone):
    """Freeze only the ResNet when the training config requests it.

    The FPN and head stay trainable, as in retinanet_dinov3.
    """
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.body = torch.nn.Linear(2, 2)
    model.backbone.fpn = torch.nn.Linear(2, 2)
    model.head = torch.nn.Linear(2, 2)
    monkeypatch.setattr(retinanet.RetinaNetHub, "from_pretrained", lambda *args, **kwargs: model)
    config = copy.deepcopy(config)
    config.train.freeze_backbone = freeze_backbone

    loaded = retinanet.Model(config).create_model(pretrained="test/checkpoint")

    assert all(p.requires_grad != freeze_backbone for p in loaded.backbone.body.parameters())
    assert all(p.requires_grad for p in loaded.backbone.fpn.parameters())
    assert all(p.requires_grad for p in loaded.head.parameters())


def test_resize_saved_with_model(tmp_path):
    """The resize setting is saved with the model and restored on load."""
    model = retinanet.RetinaNetHub(num_classes=1, label_dict={"Tree": 0}, resize=False)
    model.save_pretrained(tmp_path)

    loaded = retinanet.RetinaNetHub.from_pretrained(tmp_path)
    images, _ = loaded.eval().transform([torch.rand(3, 400, 600)])

    assert loaded.resize is False
    assert images.image_sizes == [(400, 600)]
