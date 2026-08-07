import warnings
from pathlib import Path

import torch
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN as _TorchvisionMaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from deepforest import utilities
from deepforest.model import BaseModel


class _NativeResolutionTransform(GeneralizedRCNNTransform):
    """``GeneralizedRCNNTransform`` variant whose ``resize`` is a pass-through.

    The augmentation pipeline owns all scale selection: training crops are
    taken at native ground sample distance (1024x1024 for OAM-TCD) and
    inference callers pass tiles at their own resolution. Torchvision's
    default ``min_size=800, max_size=1333`` would rescale both, changing the
    GSD the model sees in either direction. Normalize and pad-to-size-divisible
    still run as usual.

    Because resize is disabled, ``min_size`` and ``max_size`` have no effect.
    """

    def resize(self, image, target=None):
        return image, target


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
        nms_thresh: float = 0.5,
        score_thresh: float = 0.05,
        label_dict: dict = None,
        trainable_backbone_layers: int | None = None,
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
        self.inference_output = inference_output
        self.kwargs = kwargs

        if backbone_weights is not None:
            self._adjust_classes(num_classes)

        self.update_config()

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
        ``utilities.masks_to_polygons`` and replaced with a list of
        shapely polygons. Saves the per-tile output payload in
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
                    shifted["masks"] = utilities.decode_panoptic_target(shifted)
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
                output["polygons"] = utilities.masks_to_polygons(binary)

        return outputs


class Model(BaseModel):
    """DeepForest model wrapper for Mask R-CNN instance segmentation.

    Selected via ``config.architecture = "maskrcnn"`` with
    ``config.model.task`` resolving to ``"polygon"``.
    """

    def _model_kwargs(self) -> dict:
        """Constructor arguments shared by the cold-start and pretrained paths.

        ``trainable_backbone_layers`` and ``inference_output`` are
        handled by :class:`MaskRCNN` itself. Every other ``maskrcnn.*``
        field is named after a torchvision ``MaskRCNN`` argument and is
        forwarded unchanged, so adding one to the schema needs no change
        here.
        """
        torchvision_args = OmegaConf.to_container(self.config.maskrcnn, resolve=True)
        trainable_backbone_layers = torchvision_args.pop("trainable_backbone_layers")
        inference_output = torchvision_args.pop("inference_output")

        return {
            "num_classes": self.config.num_classes,
            "nms_thresh": self.config.nms_thresh,
            "score_thresh": self.config.score_thresh,
            "label_dict": dict(self.config.label_dict)
            if self.config.label_dict
            else None,
            "trainable_backbone_layers": trainable_backbone_layers,
            "inference_output": inference_output,
            "box_detections_per_img": self.config.detections_per_img,
            **torchvision_args,
        }

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
        model_kwargs = self._model_kwargs()

        if pretrained is None:
            model = MaskRCNN(backbone_weights="COCO_V1", **model_kwargs)
        else:
            model = MaskRCNN.from_pretrained(
                pretrained, revision=revision, **model_kwargs, **hf_args
            )

        return model.to(map_location)

    def check_model(self) -> None:
        """Validate the model returns instance-segmentation outputs."""
        test_model = self.create_model()
        test_model.eval()

        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        # Structure check only — no need to retain activations. See BaseModel.
        with torch.no_grad():
            predictions = test_model(x)

        assert len(predictions) == 2

        instance_key = (
            "polygons" if test_model.inference_output == "polygons" else "masks"
        )
        model_keys = sorted(predictions[1].keys())
        assert model_keys == sorted(["boxes", "labels", instance_key, "scores"])
