"""TreeFormer: multi-scale density estimation for aerial imagery.

This module replicates the architecture from the TreeFormer paper
(10.1109/TGRS.2023.3295802), which extends DM-Count (arXiv:2009.13077).
Only the supervised branch is implemented.

Two backbone families are supported:

* **PvT-V2** (default) -- ``OpenGVLab/pvt_v2_b{0-5}``.  The backbone
  natively produces 4 multi-scale feature maps at strides 4/8/16/32.

* **DINOv3 ViT** -- any HuggingFace model ID containing ``"dino"``
  (e.g. ``"facebook/dinov3-vits16-pretrain-lvd1689m"``).  A ViTDet-
  style adapter is used: the final ``last_hidden_state`` is reshaped to
  a spatial grid at stride ``patch_size``, then bilinear upsampling and
  average pooling produce the 4-scale pyramid (strides 4/8/16/32)
  expected by the Regression head.

In both cases the rest of the pipeline -- channel projection, the
multi-scale Regression head, losses and inference -- is identical.
The model predicts density maps (not raw counts) so that it transfers
to variable image sizes at test time.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from scipy.ndimage import gaussian_filter
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    PvtV2Config,
    PvtV2Model,
)

from deepforest.losses.ot_loss import OT_Loss
from deepforest.model import BaseModel
from deepforest.models.treeformer_decoder import Regression


class LayerNorm2d(nn.Module):
    """Per-position channel LayerNorm for (B, C, H, W) tensors.

    Normalises over the channel dim at each spatial position without
    permuting (avoids non-contiguous tensor issues in autograd).
    Equivalent to nn.LayerNorm([C]) applied independently per pixel.
    Implementation follows SAM / VitDet reference code.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class TreeFormerModel(nn.Module, PyTorchModelHubMixin):
    """Multi-scale density estimation model with a pluggable backbone.

    Supports two backbone families, selected by the ``backbone`` string:

    * **PvT-V2** (default) – ``"OpenGVLab/pvt_v2_b3"`` (or any ``pvt_v2_b*``
      variant).  Loaded via :class:`transformers.PvtV2Model`.  The backbone
      produces 4 multi-scale feature maps at strides 4/8/16/32.

    * **DINOv3 ViT** – any string containing ``"dino"``
      (e.g. ``"facebook/dinov3-vits16-pretrain-lvd1689m"``).  Loaded via
      :class:`transformers.AutoModel`.  A ViTDet-style adapter takes the
      final ``last_hidden_state``, reshapes patch tokens to a spatial grid,
      then generates the 4-scale pyramid (strides 4/8/16/32) via bilinear
      upsampling and average pooling to match the
      :class:`~deepforest.models.treeformer_decoder.Regression` head.
    """

    task = "keypoint"

    # Native output channel dims for each PvtV2 variant.
    HIDDEN_SIZES = {
        "pvt_v2_b0": [32, 64, 160, 256],
        "pvt_v2_b1": [64, 128, 320, 512],
        "pvt_v2_b2": [64, 128, 320, 512],
        "pvt_v2_b3": [64, 128, 320, 512],
        "pvt_v2_b4": [64, 128, 320, 512],
        "pvt_v2_b5": [64, 128, 320, 512],
    }

    # Fixed dims Regression expects
    REG_DIMS = [128, 256, 512, 1024]

    # Intermediate transformer layers to extract for the DPT-style 4-scale pyramid.
    # Early layers → fine scales (more local structure); late → coarse (more semantic).
    # ViT-G indices match DINOv3 paper §appendix Table 6.
    DINO_EXTRACT_LAYERS = {
        "vits": [3, 6, 9, 12],  # 12 layers
        "vitb": [3, 6, 9, 12],  # 12 layers (wider than S, same depth)
        "vitl": [6, 12, 18, 24],  # 24 layers
        "vitg": [10, 20, 30, 40],  # 40 layers — DINOv3 paper §appendix Table 6
    }

    def __init__(
        self,
        backbone: str = "OpenGVLab/pvt_v2_b3",
        pretrained: bool = True,
        num_classes: int = 1,
        label_dict: dict | None = None,
        num_of_iter_in_ot: int = 100,
        sinkhorn_reg: float = 1.0,
        density_sigma: float = 5.0,
        mae_weight: float = 1.0,
        ot_weight: float = 0.1,
        density_l1_weight: float = 0.01,
        count_cls_weight: float = 1.0,
        losses: list | None = None,
        norm_cood: bool = False,
        enforce_count: bool = True,
        dino_unfreeze_last_n: int = 0,
        **kwargs,
    ):
        """Initialize TreeFormerModel.

        Args:
            backbone: HuggingFace model ID.  PvT-V2 variants
                (``pvt_v2_b0`` through ``pvt_v2_b5``) and DINOv3 ViT models
                (any ID containing ``"dino"``, e.g.
                ``"facebook/dinov3-vits16-pretrain-lvd1689m"``) are supported.
            pretrained: Load pre-trained weights from the Hub.  For DINOv3
                this is always ``True``; passing ``False`` is only respected
                for PvT-V2 backbones.
            num_classes: Number of output density channels.
            label_dict: Optional mapping from class label to integer index.
            num_of_iter_in_ot: Sinkhorn iterations for OT loss.
            sinkhorn_reg: Regularisation coefficient for OT loss.
            density_sigma: Gaussian smoothing sigma for GT density maps.
            mae_weight: Weight for the count MAE loss term.
            ot_weight: Weight for the OT loss term.
            density_l1_weight: Weight for the density L1 (TV) loss term.
            count_cls_weight: Weight for the CLS-branch count loss.
            losses: Active loss terms; defaults to
                ``["count", "ot", "density_l1", "count_cls"]``.
            norm_cood: Normalise coordinates before computing OT loss.
            enforce_count: Rescale density map so its sum equals the CLS
                count prediction.
            dino_unfreeze_last_n: Number of trailing ViT transformer blocks to
                unfreeze for fine-tuning.  0 (default) keeps the entire
                backbone frozen.  Only applies to DINOv3 backbones.
        """
        super().__init__()
        self.backbone_name = backbone

        if "dino" in backbone.lower():
            self._init_dino_backbone(backbone, dino_unfreeze_last_n)
        else:
            self._init_pvt_backbone(backbone, pretrained)

        self.num_classes = num_classes
        self.label_dict = label_dict
        self.regression = Regression(num_classes=num_classes)
        self.ot_iter = num_of_iter_in_ot

        # Output stride is 4 for both backbone families: the primary density
        # map is always at 1/4 the input spatial resolution.
        self.downsample_ratio = 4

        self.sinkhorn_reg = sinkhorn_reg
        self.density_sigma = density_sigma
        self.mae_weight = mae_weight
        self.ot_weight = ot_weight
        self.density_l1_weight = density_l1_weight
        self.count_cls_weight = count_cls_weight
        self.enforce_count = enforce_count
        self.norm_cood = norm_cood
        self.dino_unfreeze_last_n = dino_unfreeze_last_n

        if losses is None:
            losses = ["count", "ot", "density_l1", "count_cls"]
        self.losses = list(losses)
        if "count_cls" not in self.losses and enforce_count:
            warnings.warn(
                "enforce_count uses the CLS branch to rescale the density map, but "
                "count_cls is not active in losses. This preserves the requested "
                "legacy behavior, but it is unsafe and can degrade spatial quality.",
                UserWarning,
                stacklevel=2,
            )
        self.active_losses = set(self.losses)

        # Losses that don't require a device are set up eagerly.
        self.density_l1 = nn.L1Loss(reduction="none")
        self.cls_l1 = nn.L1Loss()

        # OT_Loss is set up once the model device is known
        self._ot_loss: OT_Loss | None = None

        self.kwargs = kwargs
        self.update_config()

    # ------------------------------------------------------------------
    # Backbone initialisation helpers
    # ------------------------------------------------------------------

    def _init_pvt_backbone(self, backbone: str, pretrained: bool) -> None:
        """Set up PvT-V2 backbone, processor, and channel projections."""
        self.processor = AutoImageProcessor.from_pretrained(
            backbone,
            use_fast=True,
            do_normalize=True,
            do_rescale=False,
            do_resize=False,
        )

        if pretrained:
            self.backbone = PvtV2Model.from_pretrained(backbone)
        else:
            config = AutoConfig.from_pretrained(backbone)
            if not isinstance(config, PvtV2Config):
                raise TypeError(
                    f"Expected PvtV2Config for backbone {backbone}, "
                    f"got {type(config).__name__}"
                )
            self.backbone = PvtV2Model(config)

        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Suppress some noisy warnings that show in DDP.
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        for module in self.backbone.modules():
            if (
                isinstance(module, nn.Conv2d)
                and module.groups > 1
                and module.groups == module.in_channels
            ):
                module.weight.register_hook(lambda grad: grad.contiguous())

        variant = backbone.split("/")[-1]
        src = self.HIDDEN_SIZES.get(variant, None)
        if src is None:
            raise ValueError(
                f"Backbone variant {variant} isn't supported. "
                f"Please use one of {list(self.HIDDEN_SIZES.keys())}"
            )
        self.proj = nn.ModuleList(
            [nn.Conv2d(s, d, 1) for s, d in zip(src, self.REG_DIMS, strict=True)]
        )
        # PvT stride: the coarsest stage output (x3) is at stride 32.
        self.forward_stride = 32

    def _init_dino_backbone(self, backbone: str, unfreeze_last_n: int = 0) -> None:
        """Set up DINOv3 ViT backbone, processor, and channel projections.

        ViTDet-style adapter: the final ``last_hidden_state`` is reshaped
        to a spatial grid at stride ``patch_size``, then bilinear upsampling
        and average pooling generate the 4-scale pyramid (strides 4/8/16/32)
        that ``Regression`` expects.  Each scale is projected independently
        to its target channel dim via a 1x1 conv in ``self.proj``.
        """
        self.processor = AutoImageProcessor.from_pretrained(
            backbone,
            do_normalize=True,
            do_rescale=False,
            do_resize=False,
        )
        self.backbone = AutoModel.from_pretrained(backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            n_layers = self.backbone.config.num_hidden_layers
            # transformers >=5.5 wraps the layer list under `.model` (DINOv3ViTEncoder);
            # earlier versions expose it directly on the model. `.norm` is unchanged.
            blocks = (
                self.backbone.model.layer
                if hasattr(self.backbone, "model")
                else self.backbone.layer
            )
            for i in range(max(0, n_layers - unfreeze_last_n), n_layers):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        cfg = self.backbone.config
        hidden_size: int = cfg.hidden_size
        self.dino_patch_size: int = cfg.patch_size
        self.dino_num_register_tokens: int = getattr(cfg, "num_register_tokens", 0)
        self.dino_num_hidden_layers: int = cfg.num_hidden_layers

        # DPT-style multi-layer extraction: assign intermediate transformer layers
        # to the four output scales (early layers → fine scales, late → coarse),
        # mirroring DINOv3 paper §appendix Table 6.
        # Each extracted layer gets a 1×1 conv before the scale-specific proj.
        variant_key = next(
            (k for k in self.DINO_EXTRACT_LAYERS if k in backbone.lower()), None
        )
        if variant_key is not None:
            self.dino_extract_layers = self.DINO_EXTRACT_LAYERS[variant_key]
        else:
            n = self.dino_num_hidden_layers
            self.dino_extract_layers = [n // 4, n // 2, 3 * n // 4, n]
        self.layer_projs = nn.ModuleList(
            [nn.Conv2d(hidden_size, hidden_size, kernel_size=1) for _ in range(4)]
        )

        # ViTDet simple feature pyramid (Li et al. 2022, §3.1) with VitDet's
        # per-level LayerNorm + 3×3 conv + LayerNorm refinement to smooth
        # token-boundary grid artifacts.
        #   proj[0]: deconv ×4 + refine → stride patch_size/4  (finest)
        #   proj[1]: deconv ×2 + refine → stride patch_size/2
        #   proj[2]: 1×1 conv + refine  → stride patch_size     (identity)
        #   proj[3]: stride-2 conv + refine → stride patch_size×2 (coarsest)
        mid = hidden_size
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_size, mid, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(mid, self.REG_DIMS[0], kernel_size=2, stride=2),
                    LayerNorm2d(self.REG_DIMS[0]),
                    nn.Conv2d(
                        self.REG_DIMS[0], self.REG_DIMS[0], kernel_size=3, padding=1
                    ),
                    LayerNorm2d(self.REG_DIMS[0]),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_size, self.REG_DIMS[1], kernel_size=2, stride=2
                    ),
                    LayerNorm2d(self.REG_DIMS[1]),
                    nn.Conv2d(
                        self.REG_DIMS[1], self.REG_DIMS[1], kernel_size=3, padding=1
                    ),
                    LayerNorm2d(self.REG_DIMS[1]),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, self.REG_DIMS[2], kernel_size=1),
                    LayerNorm2d(self.REG_DIMS[2]),
                    nn.Conv2d(
                        self.REG_DIMS[2], self.REG_DIMS[2], kernel_size=3, padding=1
                    ),
                    LayerNorm2d(self.REG_DIMS[2]),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, self.REG_DIMS[3], kernel_size=2, stride=2),
                    LayerNorm2d(self.REG_DIMS[3]),
                    nn.Conv2d(
                        self.REG_DIMS[3], self.REG_DIMS[3], kernel_size=3, padding=1
                    ),
                    LayerNorm2d(self.REG_DIMS[3]),
                ),
            ]
        )
        # Images must be divisible by patch_size*2 so that the coarsest
        # scale (strided conv ×2 on H_p=H/patch_size) is an integer.
        self.forward_stride = self.dino_patch_size * 2

    def update_config(self):
        # Stored as config on HF
        self._hub_mixin_config = {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "label_dict": self.label_dict,
            "num_of_iter_in_ot": self.ot_iter,
            "sinkhorn_reg": self.sinkhorn_reg,
            "density_sigma": self.density_sigma,
            "mae_weight": self.mae_weight,
            "ot_weight": self.ot_weight,
            "density_l1_weight": self.density_l1_weight,
            "count_cls_weight": self.count_cls_weight,
            "losses": self.losses,
            "norm_cood": self.norm_cood,
            "enforce_count": self.enforce_count,
            "dino_unfreeze_last_n": self.dino_unfreeze_last_n,
            **self.kwargs,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_ot_loss(self) -> OT_Loss:
        """Return optimal transport loss, creating it on first call with the
        current device."""
        if self._ot_loss is None:
            self._ot_loss = OT_Loss(
                self.norm_cood,
                self.device,
                self.ot_iter,
                self.sinkhorn_reg,
            )
        return self._ot_loss

    def _normalize_density(
        self,
        score_map: torch.Tensor,
        cls_count: torch.Tensor,
        gt_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (density_map, normed_density) from raw score map and count
        scalar.

        Behaviour is controlled by ``self.enforce_count`` (set via the
        ``enforce_count`` constructor argument):

        * ``False`` (default): density_map = score_map (unconstrained
          scale). count_loss trains the spatial head directly.
        * ``True`` (physically-consistent): density_map =
          score_normed * count, so density_map.sum() == count by construction.
          OT/TV losses train the spatial distribution; count_loss trains the
          CLS branch. During training, ``gt_count`` is used when provided
          (teacher forcing): the OT gradient to the spatial head is then scaled
          by the correct GT count rather than the CLS prediction, which is noisy
          early in training. At inference ``gt_count`` is unavailable so
          ``cls_count`` is always used.

        Note: with teacher forcing, ``density_map.sum() == gt_count`` exactly,
        so ``count_loss`` (which compares density_map.sum() to gt_count) is
        trivially zero. Count learning is unaffected because ``count_cls_loss``
        still trains the CLS branch independently.

        Args:
            score_map:  (B, 1, H, W) raw non-negative output of the density head.
            cls_count:  (B,) absolute-count prediction from the CLS branch.
            gt_count:   (B,) ground-truth point counts; used during training when
                        enforce_count=True to stabilise the OT gradient scale.

        Returns:
            density_map:    (B, 1, H, W) density used for count_loss and output.
            normed_density: (B, 1, H, W) spatially normalized map (sums to 1),
                            used for OT and TV losses.
        """
        B = score_map.size(0)
        score_sum = score_map.view(B, -1).sum(1).view(B, 1, 1, 1)
        normed = score_map / (score_sum + 1e-4)
        if self.enforce_count:
            if self.training and gt_count is not None:
                count = gt_count.view(B, 1, 1, 1).clamp(min=1e-4)
            else:
                # abs() recovers valid magnitude when the CLS head goes slightly
                # negative on sparse images; clamp floors at 1e-4 to avoid
                # zero-mass density maps.
                count = cls_count.view(B, 1, 1, 1).abs().clamp(min=1e-4)
            return normed * count, normed
        return score_map, normed

    def _count_areas(self, image_shapes: list[tuple[int, int]]) -> torch.Tensor:
        """Return per-image areas for a batch, in input-pixel space.

        Used to convert between density and absolute count for the CLS
        branch.
        """
        return torch.tensor(
            [image_h * image_w for image_h, image_w in image_shapes],
            dtype=torch.float32,
            device=self.device,
        )

    def _output_shapes(
        self, image_shapes: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Return the valid output-map extent for each input image.

        This matches the existing eval-time crop behaviour, which keeps
        the stride-4 region corresponding to the unpadded image and
        ignores any batch-padding beyond it.
        """
        return [
            (
                max(image_h // self.downsample_ratio, 1),
                max(image_w // self.downsample_ratio, 1),
            )
            for image_h, image_w in image_shapes
        ]

    def _build_output_mask(
        self,
        image_shapes: list[tuple[int, int]],
        out_h: int,
        out_w: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Mask padded decoder outputs so losses ignore batch padding."""
        mask = torch.zeros(
            len(image_shapes),
            1,
            out_h,
            out_w,
            device=self.device,
            dtype=dtype,
        )
        for index, (valid_h, valid_w) in enumerate(self._output_shapes(image_shapes)):
            mask[index, :, :valid_h, :valid_w] = 1.0
        return mask

    def _cls_outputs_to_count(
        self,
        cls_output: torch.Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Return CLS-head outputs as absolute counts.

        The CLS head is trained to predict count density (count / area),
        so we multiply by each image's pixel area to recover the
        absolute count. This makes the head transfer across variable
        image sizes.
        """
        count_density = cls_output.reshape(len(image_shapes))
        return count_density * self._count_areas(image_shapes)

    def _scale_points_to_output(
        self,
        points: list,
        image_shapes: list[tuple[int, int]],
        output_shapes: list[tuple[int, int]],
    ) -> list:
        """Scale image-space point coordinates into output-map coordinates."""
        scaled_points = []
        for p, (image_h, image_w), (out_h, out_w) in zip(
            points, image_shapes, output_shapes, strict=True
        ):
            scale = torch.tensor(
                [out_w / image_w, out_h / image_h],
                dtype=torch.float32,
                device=self.device,
            )
            if len(p) == 0:
                scaled_points.append(p.clone())
            else:
                scaled_points.append(p.to(dtype=torch.float32) * scale)
        return scaled_points

    # TODO Is this correct, and even needed? Points are already tensors.
    def _cast_points(self, targets: list) -> list[torch.Tensor]:
        """Cast training targets to a list of point tensors."""
        points = []
        for target in targets:
            if isinstance(target, dict):
                point_tensor = target.get("points")
                if point_tensor is None:
                    raise ValueError("Each target dict must include a 'points' entry")
            else:
                point_tensor = target

            if not isinstance(point_tensor, torch.Tensor):
                point_tensor = torch.as_tensor(point_tensor, dtype=torch.float32)

            points.append(point_tensor.to(device=self.device, dtype=torch.float32))

        return points

    def forward_features(self, x: torch.Tensor):
        """Run backbone and project stage outputs to REG_DIMS.

        Returns:
            feats: list of 4 spatial tensors
                   ``(B, REG_DIMS[i], H_i, W_i)`` at strides 4, 8, 16, 32.
            cls:   list of 3 vectors ``(B, REG_DIMS[j])`` for j in 1..3 -
                   global average-pooled features for the CLS-branch count
                   head (substitutes PvT-v1 CLS tokens).
        """
        if "dino" in self.backbone_name.lower():
            return self._forward_features_dino(x)
        return self._forward_features_pvt(x)

    def _forward_features_pvt(self, x: torch.Tensor):
        """PvT-V2 feature extraction path."""
        out = self.backbone(x, output_hidden_states=True)
        feats = [p(h) for p, h in zip(self.proj, out.hidden_states, strict=False)]
        cls = [feats[i].mean(dim=[2, 3]) for i in range(1, 4)]
        return feats, cls

    def _forward_features_dino(self, x: torch.Tensor):
        """DINOv3 ViT feature extraction with DPT-style multi-layer adapter.

        Extracts hidden states from 4 evenly-spaced transformer layers
        (e.g. [3, 6, 9, 12] for ViT-S/12), assigns each to a scale in
        the output pyramid (early layers → fine scales, late → coarse),
        applies a per-layer 1×1 projection, then the scale-specific
        conv/deconv with VitDet-style LN+3×3+LN refinement.
        """
        out = self.backbone(x, output_hidden_states=True)
        # hidden_states[0] = patch embedding; hidden_states[k] = after block k
        hidden_states = out.hidden_states

        B, _, H, W = x.shape
        H_p = H // self.dino_patch_size
        W_p = W // self.dino_patch_size
        R = self.dino_num_register_tokens

        feats = []
        for i, layer_idx in enumerate(self.dino_extract_layers):
            tokens = hidden_states[layer_idx]  # (B, 1+R+P, D)
            patches = tokens[:, 1 + R :, :]  # (B, P, D)
            spatial = patches.permute(0, 2, 1).contiguous().reshape(B, -1, H_p, W_p)
            spatial = self.layer_projs[i](spatial)
            feats.append(self.proj[i](spatial))

        cls = [feats[i].mean(dim=[2, 3]) for i in range(1, 4)]
        return feats, cls

    def rasterize_points(
        self, im_height: int, im_width: int, points: np.ndarray
    ) -> np.ndarray:
        """Rasterize point annotations into an impulse map.

        Args:
            im_height: output map height
            im_width:  output map width
            points:    (N, 2) float array of (x, y) keypoint coordinates in
                       output-map space

        Returns:
            (im_height, im_width) float32 array with per-pixel point counts
        """
        discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
        points_np = np.asarray(points, dtype=np.float32).reshape(-1, 2)

        points_np = np.rint(points_np).astype(int)
        p_h = np.clip(points_np[:, 1], 0, im_height - 1)
        p_w = np.clip(points_np[:, 0], 0, im_width - 1)
        np.add.at(discrete_map, (p_h, p_w), 1)

        return discrete_map

    # TODO: Refactor this as points_to_density and use for dataset as well.
    def _make_gt_density(self, points: list, out_h: int, out_w: int) -> torch.Tensor:
        """Build a batched Gaussian density map (B, 1, H, W) from point
        lists."""
        sigma = self.density_sigma
        maps = []
        for _idx, p in enumerate(points):
            p_np = p.cpu().numpy()
            discrete = self.rasterize_points(out_h, out_w, p_np)
            if discrete.sum() > 0:
                smoothed = gaussian_filter(discrete, sigma=sigma)
                smoothed = smoothed * (discrete.sum() / smoothed.sum())
            else:
                smoothed = discrete
            maps.append(smoothed)
        return torch.from_numpy(np.stack(maps)).unsqueeze(1).float().to(self.device)

    def compute_loss(
        self,
        density_maps: list,
        normed_density: torch.Tensor,
        cls_outputs: list,
        targets: list,
        image_shapes: list[tuple[int, int]],
        gt_density: torch.Tensor | None = None,
    ) -> dict:
        """Compute supervised losses.

        Individual losses can be disabled via the ``losses`` constructor argument, but the default is to compute all.

        The loss function for treeformer/DM-Count follows two pathways. Some losses
        aim to train the spatial structure of the density map (optimal transport and L1).
        Others train the total count for the image (MAE count loss on the density map sum,
        and optionally an auxiliary CLS count head). We actually train to predict the
        count density (count / area) which allows for more sensible prediction on images
        with different areas so counts are scaled by the input image shape (in pixels).

        The L1 loss here is referred to as total variation (TV) loss in the DM-Count paper,
        but it's really just pixel-wise L1 between the predicted and GT density maps.
        Unlike the original TreeFormer training code, this adaptation uses a Gaussian-
        smoothed GT density target here instead of a downsampled discrete point map.

        OT loss is computed using the optimal transport implementation in losses/ot_loss.py, which
        is a PyTorch implementation of the Sinkhorn algorithm. The OT loss encourages the predicted
        density map to have its mass distributed similarly to the GT points, without enforcing
        exact pixel-wise matches. The effect is to "tighten" predictions around GT points.

        Args:
            density_maps:   [y0, y1, y2] multi-scale outputs from Regression head
            normed_density: density_maps[0] normalised to sum to 1, (B, 1, H', W')
            cls_outputs:    [yc0, yc1, yc2] count scalars from CLS-token pathway
            targets:        list of B target dicts or point tensors in image space
            image_shapes:   original (height, width) for each image in the batch
            gt_density:     optional pre-computed (B, 1, H', W') Gaussian density
                            map; computed from points if None

        Returns:
            dict with 'loss' (total scalar) plus individual named terms
        """
        density_map = density_maps[0]  # primary output, (B, 1, H', W')
        B, _, H, W = density_map.shape
        points = self._cast_points(targets)
        output_shapes = self._output_shapes(image_shapes)
        areas = self._count_areas(image_shapes)

        point_counts = torch.tensor(
            [len(p) for p in points], dtype=torch.float32, device=self.device
        )
        scaled_points = self._scale_points_to_output(points, image_shapes, output_shapes)

        if gt_density is None:
            gt_density = self._make_gt_density(scaled_points, H, W)

        active = self.active_losses
        zero = density_map.new_zeros(1)
        pred_sum = density_map.view(B, -1).sum(1)

        # Absolute-count diagnostics stay unweighted so calibration is visible
        # directly in the logs.
        count_mae = self.cls_l1(pred_sum, point_counts)
        cls_preds = torch.stack([c.reshape(B) for c in cls_outputs])  # (3, B)
        gt_counts = point_counts.unsqueeze(0).expand(3, -1)  # (3, B)

        # ---- MAE count loss -----------------------------------------------
        # log1p compresses large early-training errors, preventing the count
        # gradient from overwhelming the spatial losses (OT + density_l1).
        if "count" in active:
            count_loss = (
                self.cls_l1(torch.log1p(pred_sum), torch.log1p(point_counts))
                * self.mae_weight
            )
        else:
            count_loss = zero

        # ---- Optimal transport loss ----------------------------------------
        if "ot" in active:
            (
                ot_raw,
                ot_wd_val,
                _,
                ot_avg_its,
                ot_K_min,
                ot_beta_abs_max,
                ot_sinkhorn_err,
            ) = self._get_ot_loss()(normed_density, density_map, scaled_points)
            ot_loss = ot_raw * self.ot_weight
            # Wasserstein distance and Sinkhorn diagnostics, not used for backprop.
            ot_wd = torch.tensor(
                ot_wd_val, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_its = torch.tensor(
                ot_avg_its, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_K_min = torch.tensor(
                ot_K_min, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_beta_abs_max = torch.tensor(
                ot_beta_abs_max, device=density_map.device, dtype=torch.float32
            )
            sinkhorn_err = torch.tensor(
                ot_sinkhorn_err, device=density_map.device, dtype=torch.float32
            )
        else:
            ot_loss = zero
            ot_wd = zero
            sinkhorn_its = zero
            sinkhorn_K_min = zero
            sinkhorn_beta_abs_max = zero
            sinkhorn_err = zero

        # ---- Density L1 loss (pixel-wise L1 between normalized density maps) ----
        if "density_l1" in active:
            gt_density_normed = gt_density / (point_counts.view(B, 1, 1, 1) + 1e-4)
            per_pixel = self.density_l1(normed_density, gt_density_normed)
            density_l1_loss = (
                per_pixel.sum(dim=[1, 2, 3]).mul(point_counts).mean()
                * self.density_l1_weight
            )
        else:
            density_l1_loss = zero

        # ---- Stage GAP count regression ----
        # In density mode, keep the loss in density space: compare raw CLS
        # output (count density) to gt_count/area directly. This avoids
        # multiplying by area and then dividing it back out, which would
        # amplify gradients by ~area^2 before Adam can adapt.
        if "count_cls" in active:
            # Raw CLS predicts count density; GT must match.
            gt_counts_normed = gt_counts / areas.unsqueeze(0)
            count_cls_loss = (
                self.cls_l1(cls_preds, gt_counts_normed) * self.count_cls_weight
            )
            cls_pred_counts = cls_preds * areas.unsqueeze(0)
            count_cls_mae = self.cls_l1(cls_pred_counts, gt_counts)
        else:
            count_cls_loss = zero

        total = count_loss + ot_loss + density_l1_loss + count_cls_loss
        result = {
            "loss": total,
            "count_mae": count_mae,
            "count_loss": count_loss,
            "ot_loss": ot_loss,
            "ot_wd": ot_wd,
            "sinkhorn_its": sinkhorn_its,
            "sinkhorn_K_min": sinkhorn_K_min,
            "sinkhorn_beta_abs_max": sinkhorn_beta_abs_max,
            "sinkhorn_err": sinkhorn_err,
            "density_l1_loss": density_l1_loss,
        }
        if "count_cls" in active:
            result["count_cls_mae"] = count_cls_mae
            result["count_cls_loss"] = count_cls_loss

        return result

    def forward(
        self,
        inputs: torch.Tensor | list[torch.Tensor],
        targets: list | None = None,
        gt_density: torch.Tensor | None = None,
        gt_discrete: torch.Tensor | None = None,
    ):
        """Forward pass.

        Train mode: targets must be provided; returns a loss dict.
        Eval mode:  returns (density_map, normed_density).

        inputs: (B, C, H, W) tensor or list of (C, H, W) tensors (variable sizes).
        gt_density: optional precomputed Gaussian density target used for the
            density L1 loss during training.
        gt_discrete: deprecated alias for ``gt_density``.
        """
        if gt_density is not None and gt_discrete is not None:
            raise ValueError("Pass only one of gt_density or gt_discrete")
        if gt_density is None:
            gt_density = gt_discrete

        # Batch-pad variable-size images; record original sizes for output crop.
        if isinstance(inputs, list):
            shapes = [(img.shape[-2], img.shape[-1]) for img in inputs]
            H = max(h for h, _ in shapes)
            W = max(w for _, w in shapes)
            batch = inputs[0].new_zeros(len(inputs), inputs[0].shape[0], H, W)
            for i, img in enumerate(inputs):
                batch[i, :, : shapes[i][0], : shapes[i][1]] = img
        else:
            shapes = [(inputs.shape[2], inputs.shape[3])] * inputs.shape[0]
            batch = inputs

        # Pad to next multiple of forward_stride so that the backbone's
        # coarsest output has integer spatial dimensions.
        H, W = batch.shape[2:]
        s = self.forward_stride
        batch = F.pad(batch, (0, (s - W % s) % s, 0, (s - H % s) % s))
        padded_h, padded_w = batch.shape[2:]

        encoded = self.processor.preprocess(
            images=batch,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
        )["pixel_values"].to(self.device)

        label_feats, l_cls = self.forward_features(encoded)
        out_L, out_cls_l = self.regression(label_feats, l_cls)

        if self.training:
            if targets is None:
                raise ValueError("targets must be provided in training mode")
            output_mask = self._build_output_mask(
                shapes,
                out_L[0].shape[-2],
                out_L[0].shape[-1],
                dtype=out_L[0].dtype,
            )
            primary_counts = self._cls_outputs_to_count(out_cls_l[0], shapes)
            gt_counts = torch.tensor(
                [len(p) for p in self._cast_points(targets)],
                dtype=torch.float32,
                device=self.device,
            )
            density_map, label_normed = self._normalize_density(
                out_L[0] * output_mask,
                primary_counts,
                gt_count=gt_counts,
            )

            return self.compute_loss(
                [density_map] + out_L[1:],
                label_normed,
                out_cls_l,
                targets,
                image_shapes=shapes,
                gt_density=gt_density,
            )

        # Eval: crop each output back to its original spatial extent.
        density_list, normed_list = [], []
        for i, (valid_h, valid_w) in enumerate(self._output_shapes(shapes)):
            crop = out_L[0][i : i + 1, :, :valid_h, :valid_w].contiguous()
            cls_count = self._cls_outputs_to_count(out_cls_l[0][i : i + 1], [shapes[i]])
            dm, nd = self._normalize_density(crop, cls_count)
            density_list.append(dm)
            normed_list.append(nd)

        if isinstance(inputs, torch.Tensor):
            return torch.cat(density_list), torch.cat(normed_list)
        return density_list, normed_list


class Model(BaseModel):
    """DeepForest model wrapper for TreeFormer.

    Selected via ``config.architecture = "treeformer"``.
    """

    def create_model(
        self,
        pretrained: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **hf_args,
    ) -> TreeFormerModel:
        """Create or load a TreeFormerModel.

        Args:
            pretrained: HuggingFace repo ID to load weights from, or None.
            revision: Model revision/tag on the Hub.

        Returns:
            Configured TreeFormerModel instance.
        """
        cfg = self.config.keypoint
        label_dict = dict(self.config.label_dict) if self.config.label_dict else None
        num_classes = (
            len(label_dict) if label_dict is not None else self.config.num_classes or 1
        )

        if pretrained:
            model = TreeFormerModel.from_pretrained(
                pretrained, revision=revision, **hf_args
            )
        else:
            model = TreeFormerModel(
                num_classes=num_classes,
                label_dict=label_dict,
                num_of_iter_in_ot=cfg.num_of_iter_in_ot,
                sinkhorn_reg=cfg.sinkhorn_reg,
                density_sigma=cfg.density_sigma,
                mae_weight=cfg.mae_weight,
                ot_weight=cfg.ot_weight,
                density_l1_weight=cfg.density_l1_weight,
                count_cls_weight=cfg.count_cls_weight,
                losses=list(cfg.losses) if cfg.losses is not None else None,
                norm_cood=cfg.norm_cood,
                enforce_count=cfg.enforce_count,
                backbone=cfg.backbone,
                dino_unfreeze_last_n=cfg.dino_unfreeze_last_n,
                **hf_args,
            )

        return model.to(map_location)
