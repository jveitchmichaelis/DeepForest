"""Class-balancing helpers for instance-segmentation / detection training.

Two complementary strategies for combating per-class imbalance:

- :func:`build_class_balanced_sampler` upweights rare-class images at sample
  time via a :class:`torch.utils.data.WeightedRandomSampler`.
- :func:`compute_class_loss_weights` produces inverse-frequency weights for
  the Fast R-CNN classifier CE.

Both consume the same annotation DataFrame (``image_path``, ``label`` columns)
so the two interventions can be ablated independently.
"""

import pandas as pd
from torch.utils.data import WeightedRandomSampler


def class_counts(annotations: pd.DataFrame) -> dict[str, int]:
    """Per-class instance counts from an annotation DataFrame."""
    return annotations["label"].value_counts().to_dict()


def build_class_balanced_sampler(
    annotations: pd.DataFrame, image_names: list[str] | None = None
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that upweights rare-class images.

    Per-image weight = ``max_c (1 / instance_count[c])`` for classes present
    in the image. An image containing a rare class is drawn ~ ``N_majority /
    N_rare`` times more often than a majority-only image.

    Args:
        annotations: DataFrame with ``image_path`` and ``label`` columns
            (as produced by ``utilities.read_file``).
        image_names: Optional explicit ordering of image paths. If omitted,
            uses ``annotations["image_path"].unique()``. Must match the
            dataset's ``__getitem__`` ordering for the sampler indices to
            line up with the dataset.
    """
    counts = class_counts(annotations)
    inv = {c: 1.0 / n for c, n in counts.items()}
    if image_names is None:
        image_names = annotations["image_path"].unique()
    labels_per_image = annotations.groupby("image_path")["label"].unique().to_dict()
    weights = [
        max((inv[c] for c in labels_per_image.get(name, [])), default=0.0)
        for name in image_names
    ]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_class_loss_weights(
    annotations: pd.DataFrame, label_dict: dict[str, int], num_classes: int
) -> list[float]:
    """Inverse-frequency class weights for Fast R-CNN's classifier CE.

    ``weight[c+1] = N_total / ((num_classes + 1) * N_c)`` for each foreground
    class ``c`` in ``label_dict``; ``weight[0]`` (background) is fixed at
    ``1.0``. Index 0 is background to match torchvision's label convention,
    so foreground class indices are offset by +1 relative to ``label_dict``.

    Args:
        annotations: DataFrame with ``label`` column.
        label_dict: Mapping of string label → zero-indexed foreground id
            (as used by DeepForest's API).
        num_classes: Number of foreground classes.

    Returns:
        Weight vector of length ``num_classes + 1``.
    """
    counts = class_counts(annotations)
    total = sum(counts.values())
    weights = [1.0] * (num_classes + 1)
    for name, idx in label_dict.items():
        n_c = counts.get(name, 0)
        if n_c > 0:
            weights[idx + 1] = total / ((num_classes + 1) * n_c)
    return weights
