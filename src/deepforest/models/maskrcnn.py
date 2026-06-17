import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models.detection import roi_heads as _rh
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN as _TorchvisionMaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from deepforest.model import BaseModel

# Module-global class weights for the Fast R-CNN classifier. Set via
# set_loss_class_weights(); when None, the original torchvision loss runs.
_CLASS_WEIGHTS: torch.Tensor | None = None
_ORIG_FASTRCNN_LOSS = _rh.fastrcnn_loss


def _weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """Monkey-patched replacement for torchvision.models.detection.roi_heads.fastrcnn_loss.

    Identical to the original except the classification CE is weighted by
    ``_CLASS_WEIGHTS`` when set. The box regression loss is untouched.
    """
    if _CLASS_WEIGHTS is None:
        return _ORIG_FASTRCNN_LOSS(
            class_logits, box_regression, labels, regression_targets
        )

    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    weights = _CLASS_WEIGHTS.to(class_logits.device, dtype=class_logits.dtype)
    classification_loss = F.cross_entropy(class_logits, labels_cat, weight=weights)

    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_inds_subset]
    N, _ = class_logits.shape
    box_regression_r = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    box_loss = (
        F.smooth_l1_loss(
            box_regression_r[sampled_pos_inds_subset, labels_pos],
            regression_targets_cat[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        / labels_cat.numel()
    )

    return classification_loss, box_loss


_rh.fastrcnn_loss = _weighted_fastrcnn_loss


def set_loss_class_weights(weights: list[float] | torch.Tensor | None) -> None:
    """Set the per-class weights used by Fast R-CNN's classification CE.

    Args:
        weights: Tensor or list of length ``num_classes + 1`` where index 0
            is background. Pass ``None`` to disable weighting.
    """
    global _CLASS_WEIGHTS
    if weights is None:
        _CLASS_WEIGHTS = None
    else:
        _CLASS_WEIGHTS = torch.as_tensor(weights, dtype=torch.float32)


def clear_loss_class_weights() -> None:
    """Disable Fast R-CNN classifier CE weighting (restores torchvision default)."""
    set_loss_class_weights(None)


class MaskRCNN(_TorchvisionMaskRCNN, PyTorchModelHubMixin):
    """Mask R-CNN extension that allows the use of the HF Hub API.

    DeepForest labels are zero-indexed foreground classes (e.g.
    ``{"Tree": 0}``). torchvision detection models reserve class ``0``
    for background, so this wrapper builds the underlying model with
    ``num_classes + 1`` outputs and transparently shifts labels by one:
    targets are shifted up before training and predictions are shifted
    back down. Callers therefore always see zero-indexed labels,
    matching the box and point workflows.
    """

    task: str = "polygon"

    def __init__(
        self,
        backbone_weights: str | None = None,
        num_classes: int = 1,
        nms_thresh: float = 0.05,
        score_thresh: float = 0.5,
        label_dict: dict = None,
        trainable_backbone_layers: int | None = None,
        **kwargs,
    ):
        backbone_kwargs = {"weights": backbone_weights}
        if trainable_backbone_layers is not None:
            backbone_kwargs["trainable_backbone_layers"] = trainable_backbone_layers
        backbone = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            **backbone_kwargs
        ).backbone

        # torchvision reserves class 0 for background, so add one class.
        super().__init__(
            backbone=backbone,
            num_classes=num_classes + 1,
            box_nms_thresh=nms_thresh,
            box_score_thresh=score_thresh,
            **kwargs,
        )

        self.num_classes = num_classes
        self.label_dict = label_dict
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.trainable_backbone_layers = trainable_backbone_layers
        self.kwargs = kwargs

        self.update_config()

    def apply_class_balanced_loss(self, annotations, label_dict: dict[str, int]) -> None:
        """Enable inverse-frequency classifier-CE weighting from train annotations.

        Args:
            annotations: DataFrame with a ``label`` column (e.g. produced by
                ``utilities.read_file``).
            label_dict: Mapping of string label → zero-indexed foreground id.
        """
        from deepforest.datasets.sampling import compute_class_loss_weights

        weights = compute_class_loss_weights(annotations, label_dict, self.num_classes)
        set_loss_class_weights(weights)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *, num_classes=None, label_dict=None, **kwargs
    ):
        """Override default from_pretrained to support changing the number of
        classes in a pretrained model.

        If the target num_classes differs from the model's num_classes,
        the box and mask heads are reinitialized to compensate.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Override class info if specified
        if num_classes is not None and label_dict is not None:
            if len(label_dict) != num_classes:
                raise ValueError(
                    f"num_classes ({num_classes}) does not match the number of labels "
                    f"in label_dict ({len(label_dict)})."
                )

            if num_classes != model.num_classes:
                warnings.warn(
                    f"The number of classes in your config differs "
                    f"compared to the model checkpoint ({model.num_classes}-class)."
                    f" If you are fine-tuning on a new dataset that "
                    f"has {num_classes} then this is expected.",
                    stacklevel=2,
                )

                model._adjust_classes(num_classes)

            model.label_dict = label_dict
            model.update_config()

        return model

    def _adjust_classes(self, num_classes):
        """Rebuild the box and mask predictor heads for ``num_classes``
        foreground classes (``num_classes + 1`` with background)."""
        self.num_classes = num_classes

        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.roi_heads.mask_predictor.conv5_mask.out_channels
        self.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes + 1
        )

    def update_config(self):
        # Stored as config on HF
        self._hub_mixin_config = {
            "num_classes": self.num_classes,
            "nms_thresh": self.nms_thresh,
            "score_thresh": self.score_thresh,
            "label_dict": self.label_dict,
            "trainable_backbone_layers": self.trainable_backbone_layers,
            **self.kwargs,
        }

    def forward(self, images, targets=None):
        """Shift labels between DeepForest (0-indexed) and torchvision
        (background=0) conventions.

        In training mode, target labels are shifted up by one. In eval
        mode, predicted labels are shifted back down by one.
        """
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            shifted_targets = []
            for target in targets:
                shifted = dict(target)
                shifted["labels"] = target["labels"] + 1
                shifted_targets.append(shifted)
            return super().forward(images, shifted_targets)

        outputs = super().forward(images)
        for output in outputs:
            output["labels"] = output["labels"] - 1
        return outputs


class Model(BaseModel):
    """DeepForest model wrapper for Mask R-CNN instance segmentation.

    Selected via ``config.architecture = "maskrcnn"`` with
    ``config.model.task`` resolving to ``"polygon"``.
    """

    def create_model(
        self,
        pretrained: str | Path | None = None,
        *,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> MaskRCNN:
        """Create a Mask R-CNN model.

        Args:
            pretrained: If supplied, repository ID for weight download, otherwise use default COCO backbone weights
            revision: Repository revision
            map_location: Device to load weights onto
            **hf_args: Any other arguments to load_pretrained
        Returns:
            model: a pytorch nn module
        """
        label_dict = dict(self.config.label_dict) if self.config.label_dict else None
        mrcnn = self.config.maskrcnn

        if pretrained is None:
            model = MaskRCNN(
                backbone_weights="COCO_V1",
                num_classes=self.config.num_classes,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                label_dict=label_dict,
                trainable_backbone_layers=mrcnn.trainable_backbone_layers,
                box_detections_per_img=self.config.detections_per_img,
                rpn_pre_nms_top_n_test=mrcnn.rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_test=mrcnn.rpn_post_nms_top_n_test,
            )
        else:
            model = MaskRCNN.from_pretrained(
                pretrained,
                revision=revision,
                num_classes=self.config.num_classes,
                label_dict=label_dict,
                nms_thresh=self.config.nms_thresh,
                score_thresh=self.config.score_thresh,
                trainable_backbone_layers=mrcnn.trainable_backbone_layers,
                box_detections_per_img=self.config.detections_per_img,
                rpn_pre_nms_top_n_test=mrcnn.rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_test=mrcnn.rpn_post_nms_top_n_test,
                **hf_args,
            )

        return model.to(map_location)

    def check_model(self) -> None:
        """Validate the model returns instance-segmentation outputs."""
        test_model = self.create_model()
        test_model.eval()

        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = test_model(x)
        assert len(predictions) == 2

        model_keys = sorted(predictions[1].keys())
        assert model_keys == ["boxes", "labels", "masks", "scores"]
