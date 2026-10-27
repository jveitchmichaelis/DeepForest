# test RetinaNet with a DINOv3 backbone
import pytest
import torch

from deepforest import main, utilities
from deepforest.models import retinanet_dinov3

# Smallest DINOv3 model - a Hugging Face token with access is needed.
DINOV3_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"


@pytest.fixture(scope="module")
def dinov3_model():
    return retinanet_dinov3.RetinaNetDinov3Hub(
        backbone_weights=DINOV3_REPO, label_dict={"Tree": 0}
    )


@pytest.fixture()
def dinov3_config():
    return utilities.load_config("dinov3", overrides={"model": {"name": DINOV3_REPO}})


def test_inference(dinov3_model):
    # Test inference produces some preds with an untrained model.
    dinov3_model.eval()
    with torch.no_grad():
        predictions = dinov3_model([torch.rand(3, 400, 400), torch.rand(3, 300, 350)])
    assert len(predictions) == 2
    for pred in predictions:
        assert {"boxes", "scores", "labels"} <= pred.keys()


def test_train(dinov3_model):
    # Test training produces nonzero loss.
    dinov3_model.train()
    targets = [{"boxes": torch.tensor([[203., 67, 227, 90], [256, 99, 288, 140]]),
                "labels": torch.tensor([0, 0])}]
    loss_dict = dinov3_model([torch.rand(3, 400, 400)], targets)
    assert loss_dict["bbox_regression"] > 0
    assert loss_dict["classification"] > 0


def test_all_parameters_used(dinov3_model):
    # DDP fails if a trainable parameter gets no gradient
    dinov3_model.train()
    dinov3_model.zero_grad()
    targets = [{"boxes": torch.tensor([[203., 67, 227, 90]]), "labels": torch.tensor([0])}]
    sum(dinov3_model([torch.rand(3, 400, 400)], targets).values()).backward()
    unused = [name for name, p in dinov3_model.named_parameters()
              if p.requires_grad and p.grad is None]
    assert unused == []


def test_freeze_backbone():
    model = retinanet_dinov3.RetinaNetDinov3Hub(
        backbone_weights=DINOV3_REPO, freeze_backbone=True
    )
    # Only the transformer is frozen, the feature pyramid is trained
    assert not any(p.requires_grad for p in model.backbone.model.parameters())
    assert all(p.requires_grad for p in model.backbone.fpn_1x.parameters())


def test_from_encoder():
    model = retinanet_dinov3.RetinaNetDinov3Hub.from_pretrained(
        DINOV3_REPO, num_classes=1, label_dict={"Tree": 0}
    )
    assert model.backbone.repo_id == DINOV3_REPO


def test_from_encoder_needs_labels():
    with pytest.raises(ValueError, match="Set num_classes and label_dict"):
        retinanet_dinov3.RetinaNetDinov3Hub.from_pretrained(DINOV3_REPO)


def test_wrong_checkpoint():
    # A ResNet50 RetinaNet is not a retinanet_dinov3 checkpoint
    with pytest.raises(ValueError, match="not a DINOv3 encoder"):
        retinanet_dinov3.RetinaNetDinov3Hub.from_pretrained("weecology/deepforest-tree")


def test_config_from_encoder(dinov3_config):
    m = main.deepforest(config=dinov3_config)
    assert isinstance(m.model, retinanet_dinov3.RetinaNetDinov3Hub)
    assert m.model.backbone.repo_id == DINOV3_REPO
    assert m.label_dict == {"Tree": 0}
