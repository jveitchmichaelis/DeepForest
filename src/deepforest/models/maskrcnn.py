import warnings
from pathlib import Path

import cv2
import numpy as np
import shapely.geometry
import torch
import torch.nn.functional as F
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection import roi_heads as _rh
from torchvision.models.detection import rpn as _rpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN as _TorchvisionMaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from deepforest.model import BaseModel

# Module-global class weights for the Fast R-CNN classifier. Set via
# set_loss_class_weights(); when None, the original torchvision loss runs.
_CLASS_WEIGHTS: torch.Tensor | None = None
_ORIG_FASTRCNN_LOSS = _rh.fastrcnn_loss
_ORIG_RPN_COMPUTE_LOSS = _rpn.RegionProposalNetwork.compute_loss

# Detectron2 uses pure L1 (beta=0) for both the RPN and the second-stage
# box regression. Torchvision hardcodes ``beta=1/9`` in two places. With
# small-std init, the predictor starts in the smooth-L1 quadratic region
# where the gradient is ``9x`` for ``|x| < 1/9`` — stronger than pure
# L1's constant ±1 cap, which can drive box-reg divergence under SGD.
# Matches ``MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0`` and
# ``MODEL.RPN.SMOOTH_L1_BETA = 0.0`` in the Restor-Foundation/tcd config.
_SMOOTH_L1_BETA: float = 0.0


def _weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """Monkey-patched replacement for ``roi_heads.fastrcnn_loss``.

    Two changes vs. the upstream version:

    1. When ``_CLASS_WEIGHTS`` is set, the classification cross-entropy
       is weighted (see ``set_loss_class_weights``).
    2. The smooth-L1 box-regression beta is taken from
       ``_SMOOTH_L1_BETA`` rather than hardcoded ``1/9``, so it can be
       set to ``0`` (pure L1) to match Detectron2's default.
    """
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    if _CLASS_WEIGHTS is None:
        classification_loss = F.cross_entropy(class_logits, labels_cat)
    else:
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
            beta=_SMOOTH_L1_BETA,
            reduction="sum",
        )
        / labels_cat.numel()
    )

    return classification_loss, box_loss


def _rpn_compute_loss(
    self,
    objectness: Tensor,
    pred_bbox_deltas: Tensor,
    labels: list[Tensor],
    regression_targets: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Monkey-patched replacement for ``RegionProposalNetwork.compute_loss``.

    Identical to the upstream method except the smooth-L1 ``beta`` is
    taken from the module-level ``_SMOOTH_L1_BETA`` so we can match
    Detectron2's pure-L1 RPN box regression.
    """
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)

    box_loss = (
        F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets_cat[sampled_pos_inds],
            beta=_SMOOTH_L1_BETA,
            reduction="sum",
        )
        / sampled_inds.numel()
    )

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels_cat[sampled_inds]
    )

    return objectness_loss, box_loss


_rh.fastrcnn_loss = _weighted_fastrcnn_loss
_rpn.RegionProposalNetwork.compute_loss = _rpn_compute_loss


class _CheckpointedLayer(torch.nn.Module):
    """Wrap an nn.Module so its forward runs under activation checkpointing."""

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # use_reentrant=False is required for inputs without requires_grad
        # (e.g. the first ResNet layer's input feature map under AMP).
        return checkpoint(self.layer, x, use_reentrant=False)


class _NativeResolutionTransform(GeneralizedRCNNTransform):
    """``GeneralizedRCNNTransform`` variant whose ``resize`` is a pass-through.

    Training crops are produced at the augmentation-pipeline target size
    (640×640 for OAM-TCD) and inference callers pass tiles at native
    resolution. Torchvision's default ``min_size=800, max_size=1333``
    would upscale 640 crops to 800 at training (wasted compute) and
    downscale large patches at inference (destructive resolution loss).
    Disabling resize matches Detectron2's ``MIN_SIZE_TEST=0`` and lets
    the augmentation pipeline own all scale selection upstream. Normalize
    and pad-to-size-divisible still run as usual.
    """

    def resize(self, image, target=None):
        return image, target


def _decode_panoptic_target(
    target: dict, dtype: torch.dtype = torch.uint8
) -> torch.Tensor:
    """Decode a panoptic-encoded target into per-instance binary masks.

    Returns a ``(N, H, W)`` tensor on the same device as ``panoptic_masks``.
    ``N`` is the number of surviving instance IDs in ``unique_ids``.
    """
    panoptic = target["panoptic_masks"]
    ids = target["unique_ids"]
    H, W = panoptic.shape
    if ids.numel() == 0:
        return torch.zeros((0, H, W), dtype=dtype, device=panoptic.device)
    return (panoptic.unsqueeze(0) == ids.view(-1, 1, 1)).to(dtype)


def _masks_to_polygons(masks: np.ndarray) -> list[shapely.geometry.Polygon]:
    """Vectorise a stack of binary masks via the largest external contour.

    Mirrors ``utilities.mask_to_polygon`` but operates on a stack; kept
    in the model module so ``MaskRCNN.forward`` doesn't pull in a
    cycle-prone import from ``utilities``.
    """
    polygons: list[shapely.geometry.Polygon] = []
    for mask in masks:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            polygons.append(shapely.geometry.Polygon())
            continue
        largest = max(contours, key=cv2.contourArea)
        coords = largest.reshape(-1, 2)
        if coords.shape[0] < 3:
            polygons.append(shapely.geometry.Polygon())
            continue
        polygon = shapely.geometry.Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygons.append(polygon)
    return polygons


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
        gradient_checkpointing: bool = False,
        inference_output: str = "dense",
        **kwargs,
    ):
        if inference_output not in ("dense", "polygons"):
            raise ValueError(
                f"inference_output must be 'dense' or 'polygons', got {inference_output!r}"
            )
        factory_kwargs = {"weights": backbone_weights}
        if trainable_backbone_layers is not None:
            factory_kwargs["trainable_backbone_layers"] = trainable_backbone_layers
        # Build the full torchvision Mask R-CNN. When ``backbone_weights``
        # is set (e.g. ``"COCO_V1"``) this gives us COCO-pretrained
        # weights for backbone + FPN + RPN + box_head + mask_head +
        # predictors. We use the factory's backbone module directly and
        # copy the rest of the weights via ``load_state_dict`` after
        # ``super().__init__`` builds our architectural shell.

        pretrained_full = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            **factory_kwargs
        )

        # Reuse the factory's RPN / box / mask head modules so the internal
        # architecture matches the pretrained state_dict shape.
        head_kwargs = {
            "rpn_anchor_generator": pretrained_full.rpn.anchor_generator,
            "rpn_head": pretrained_full.rpn.head,
            "box_head": pretrained_full.roi_heads.box_head,
            "mask_head": pretrained_full.roi_heads.mask_head,
        }

        if backbone_weights is None:
            # Cold start — no COCO weights to preserve.
            super().__init__(
                backbone=pretrained_full.backbone,
                num_classes=num_classes + 1,
                box_nms_thresh=nms_thresh,
                box_score_thresh=score_thresh,
                **head_kwargs,
                **kwargs,
            )
        else:
            # Build with the pretrained model's num_classes so state_dict
            # loads cleanly, then re-shape final predictor layers for our
            # num_classes via ``_adjust_classes``. This preserves the
            # COCO-pretrained RPN, box_head, mask_head, and the
            # *non-final-layer* predictor weights — only the last cls /
            # bbox / mask channels are re-init'd.
            pretrained_state = pretrained_full.state_dict()
            coco_num_classes_ext = (
                pretrained_full.roi_heads.box_predictor.cls_score.out_features
            )
            super().__init__(
                backbone=pretrained_full.backbone,
                num_classes=coco_num_classes_ext,
                box_nms_thresh=nms_thresh,
                box_score_thresh=score_thresh,
                **head_kwargs,
                **kwargs,
            )
            self.load_state_dict(pretrained_state)

        # Replace the GeneralizedRCNNTransform with a no-resize version so
        # the model runs at the size produced by the dataset / caller. See
        # ``_NativeResolutionTransform`` for rationale.
        prev_transform = self.transform
        self.transform = _NativeResolutionTransform(
            min_size=prev_transform.min_size,
            max_size=prev_transform.max_size,
            image_mean=prev_transform.image_mean,
            image_std=prev_transform.image_std,
            size_divisible=prev_transform.size_divisible,
            fixed_size=prev_transform.fixed_size,
        )

        self.num_classes = num_classes
        self.label_dict = label_dict
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.trainable_backbone_layers = trainable_backbone_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.inference_output = inference_output
        self.kwargs = kwargs

        # Gradient checkpointing on the trainable ResNet stages. layer1 is
        # frozen under trainable_backbone_layers=3 (the OAM-TCD setting),
        # so it stores no activations and the wrapper buys nothing — skip.
        if gradient_checkpointing:
            body = self.backbone.body
            for name in ("layer2", "layer3", "layer4"):
                if hasattr(body, name):
                    setattr(body, name, _CheckpointedLayer(getattr(body, name)))

        # Detectron2-style init for the box predictor. Torchvision's
        # ``FastRCNNPredictor`` falls back to ``nn.Linear``'s default
        # Kaiming-uniform init (~30x larger std than Detectron2 uses for
        # ``bbox_pred``); under SGD with momentum that destabilizes the
        # box regression early in training and the val box_reg loss
        # diverges. The mask predictor already uses Kaiming-normal
        # ``fan_out`` in torchvision so it doesn't need adjustment.
        #
        # When we loaded COCO weights, ``_adjust_classes`` will re-init
        # the predictor for our ``num_classes`` and call this internally.
        if backbone_weights is None:
            self._init_box_predictor()
        else:
            self._adjust_classes(num_classes)

        self.update_config()

    def _init_box_predictor(self) -> None:
        """Re-init the box predictor with Detectron2's small-std recipe."""
        box_predictor = self.roi_heads.box_predictor
        torch.nn.init.normal_(box_predictor.cls_score.weight, std=0.01)
        torch.nn.init.normal_(box_predictor.bbox_pred.weight, std=0.001)
        torch.nn.init.constant_(box_predictor.cls_score.bias, 0)
        torch.nn.init.constant_(box_predictor.bbox_pred.bias, 0)

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
        self._init_box_predictor()

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
            "gradient_checkpointing": self.gradient_checkpointing,
            "inference_output": self.inference_output,
            **self.kwargs,
        }

    def forward(self, images, targets=None):
        """Run the underlying torchvision Mask R-CNN, handling DeepForest's
        zero-indexed label convention and (training-time) panoptic targets.

        Targets coming from :class:`PolygonDataset` carry
        ``panoptic_masks`` + ``unique_ids`` instead of dense
        ``(N, H, W)`` masks. In training mode we decode them to dense
        masks on-device just before the parent forward — the
        ``(N, H, W)`` tensor never lives on CPU.

        In eval mode and when ``inference_output == "polygons"``, the
        dense mask logits returned by the model are vectorised via
        ``cv2.findContours`` and replaced with a list of shapely
        polygons. Saves the per-tile output payload in
        ``predict_tile`` from ~GB to ~MB.
        """
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            shifted_targets = []
            for target in targets:
                shifted = dict(target)
                shifted["labels"] = target["labels"] + 1
                if "masks" not in shifted and "panoptic_masks" in shifted:
                    shifted["masks"] = _decode_panoptic_target(shifted)
                    shifted.pop("panoptic_masks", None)
                    shifted.pop("unique_ids", None)
                shifted_targets.append(shifted)
            return super().forward(images, shifted_targets)

        outputs = super().forward(images)
        for output in outputs:
            output["labels"] = output["labels"] - 1

        if self.inference_output == "polygons":
            for output in outputs:
                dense = output.pop("masks", None)
                if dense is None or dense.numel() == 0:
                    output["polygons"] = []
                    continue
                binary = (dense.squeeze(1) > 0.5).cpu().numpy()
                output["polygons"] = _masks_to_polygons(binary)

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
                gradient_checkpointing=mrcnn.gradient_checkpointing,
                inference_output=mrcnn.inference_output,
                box_detections_per_img=self.config.detections_per_img,
                rpn_pre_nms_top_n_train=mrcnn.rpn_pre_nms_top_n_train,
                rpn_post_nms_top_n_train=mrcnn.rpn_post_nms_top_n_train,
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
                gradient_checkpointing=mrcnn.gradient_checkpointing,
                inference_output=mrcnn.inference_output,
                box_detections_per_img=self.config.detections_per_img,
                rpn_pre_nms_top_n_train=mrcnn.rpn_pre_nms_top_n_train,
                rpn_post_nms_top_n_train=mrcnn.rpn_post_nms_top_n_train,
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

        instance_key = (
            "polygons" if test_model.inference_output == "polygons" else "masks"
        )
        model_keys = sorted(predictions[1].keys())
        assert model_keys == sorted(["boxes", "labels", instance_key, "scores"])
