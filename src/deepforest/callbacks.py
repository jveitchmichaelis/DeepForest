"""DeepForest callback for logging images during training.

Callbacks must implement on_epoch_begin, on_epoch_end, on_fit_end,
on_fit_begin methods and inject model and epoch kwargs.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shapely.geometry
import supervision as sv
import torch
from PIL import Image
from pytorch_lightning import Callback

from deepforest import utilities, visualize
from deepforest.datasets.training import BoxDataset


def _df_to_comet_annotations(
    df: pd.DataFrame | None, layer_name: str, label_dict: dict | None = None
) -> dict | None:
    """Convert an annotation DataFrame to a Comet annotation layer.

    Each row's shapely ``geometry`` becomes either a ``boxes`` entry (for
    axis-aligned rectangles) or a ``points`` entry (flattened polygon
    coordinates). Returns ``None`` when there is nothing to log so the
    caller can drop empty layers.
    """
    if df is None or len(df) == 0 or "geometry" not in df.columns:
        return None

    inv_label = {idx: name for name, idx in label_dict.items()} if label_dict else None

    items = []
    for row in df.itertuples(index=False):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        label = getattr(row, "label", None)
        if inv_label is not None and label in inv_label:
            label = inv_label[label]
        label = str(label) if label is not None else layer_name

        entry: dict = {"label": label}
        if hasattr(row, "score") and row.score is not None and not pd.isna(row.score):
            entry["score"] = round(float(row.score), 2)

        if isinstance(geom, shapely.geometry.Point):
            # Render points as a small fixed-size marker box.
            half = 5.0
            entry["boxes"] = [
                [float(geom.x) - half, float(geom.y) - half, 2 * half, 2 * half]
            ]
        elif isinstance(geom, shapely.geometry.Polygon | shapely.geometry.MultiPolygon):
            # Take the bbox of the full geometry so MultiPolygons keep all components.
            xmin, ymin, xmax, ymax = geom.bounds
            entry["boxes"] = [
                [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
            ]
            # For polygons, also emit the outline. Skip when it's exactly the
            # bbox (axis-aligned rectangle) to avoid double-rendering.
            # For MultiPolygons, use the largest piece's outline.
            poly = (
                max(geom.geoms, key=lambda g: g.area)
                if isinstance(geom, shapely.geometry.MultiPolygon)
                else geom
            )
            if poly.area < poly.envelope.area:
                coords = np.asarray(poly.exterior.coords, dtype=float)
                if len(coords) >= 4:
                    entry["points"] = [coords[:-1].flatten().tolist()]
        else:
            continue

        items.append(entry)

    if not items:
        return None

    return {"name": layer_name, "data": items}


class MemoryMonitorCallback(Callback):
    """Log per-batch process and CUDA memory to all attached loggers.

    Emits four scalars (in GB) at every ``every_n_batches`` train and
    validation batch:

    - ``mem/{phase}_main_gb``: main process RSS
    - ``mem/{phase}_workers_gb``: summed RSS of all child processes (the
      DataLoader workers)
    - ``mem/{phase}_cuda_alloc_gb``: ``torch.cuda.memory_allocated()``
    - ``mem/{phase}_cuda_peak_gb``: ``torch.cuda.max_memory_allocated()``
      since the previous emit (peak is reset after logging)

    Used to localize OOMs: a growing ``workers_gb`` indicates dataset /
    augmentation memory blow-up; a growing ``main_gb`` points at metric
    accumulation or logger buffering; a CUDA peak that climbs each
    epoch suggests allocator fragmentation.
    """

    def __init__(self, every_n_batches: int = 10):
        super().__init__()
        self.every_n_batches = every_n_batches
        self._process: psutil.Process | None = None

    def _proc(self) -> psutil.Process:
        if self._process is None:
            self._process = psutil.Process()
        return self._process

    def _emit(self, trainer, phase: str) -> None:
        if not trainer.loggers:
            return
        try:
            p = self._proc()
            main_gb = p.memory_info().rss / 1e9
            children_gb = (
                sum(c.memory_info().rss for c in p.children(recursive=True)) / 1e9
            )
        except psutil.Error:
            return

        metrics = {
            f"mem/{phase}_main_gb": main_gb,
            f"mem/{phase}_workers_gb": children_gb,
        }
        if torch.cuda.is_available():
            metrics[f"mem/{phase}_cuda_alloc_gb"] = (
                torch.cuda.memory_allocated() / 1e9
            )
            metrics[f"mem/{phase}_cuda_peak_gb"] = (
                torch.cuda.max_memory_allocated() / 1e9
            )
            torch.cuda.reset_peak_memory_stats()

        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_batches == 0:
            self._emit(trainer, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx % self.every_n_batches == 0:
            self._emit(trainer, "val")


class ImagesCallback(Callback):
    """Log evaluation images during training.

    Args:
        save_dir: Directory to save predicted images
        n: Number of images to process
        every_n_epochs: Run interval in epochs
        select_random: Whether to select random images
        color: Bounding box color as BGR tuple
        thickness: Border line thickness in pixels
    """

    def __init__(
        self,
        save_dir,
        prediction_samples=2,
        dataset_samples=5,
        every_n_epochs=5,
        select_random=False,
        color=None,
        thickness=2,
    ):
        self.savedir = save_dir
        self.prediction_samples = prediction_samples
        self.dataset_samples = dataset_samples
        self.color = color
        self.thickness = thickness
        self.select_random = select_random
        self.every_n_epochs = every_n_epochs

    def on_train_start(self, trainer, pl_module):
        """Log sample images from training and validation datasets at training
        start."""
        if trainer.fast_dev_run:
            return

        self.trainer = trainer
        self.pl_module = pl_module

        # Training samples
        pl_module.print("Logging training dataset samples.")
        train_ds = trainer.train_dataloader.dataset
        self._log_dataset_sample(train_ds, split="train")

        # Validation samples
        if trainer.val_dataloaders:
            pl_module.print("Logging validation dataset samples.")
            val_ds = trainer.val_dataloaders.dataset
            self._log_dataset_sample(val_ds, split="validation")

    def on_validation_end(self, trainer, pl_module):
        """Run callback at validation end."""
        if trainer.sanity_checking or trainer.fast_dev_run:
            return

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.print("Logging prediction samples")
            self._log_last_predictions(trainer, pl_module)

    def _log_dataset_sample(self, dataset: BoxDataset, split: str):
        """Log random samples from a DeepForest BoxDataset."""
        if self.dataset_samples == 0:
            return

        out_dir = os.path.join(self.savedir, split + "_sample")
        os.makedirs(out_dir, exist_ok=True)
        n_samples = min(self.dataset_samples, len(dataset))
        sample_indices = torch.randperm(len(dataset))[:n_samples]

        sample_data = [dataset[int(idx)] for idx in sample_indices]
        sample_images = [data[0] for data in sample_data]
        sample_targets = [data[1] for data in sample_data]
        sample_paths = [data[2] for data in sample_data]

        for image, target, path in zip(
            sample_images, sample_targets, sample_paths, strict=False
        ):
            image_annotations = target.copy()
            image_annotations = utilities.format_geometry(image_annotations, scores=False)

            basename = Path(path).stem
            image = (255 * image.cpu().numpy().transpose((1, 2, 0))).astype(np.uint8)
            out_path = os.path.join(out_dir, basename + ".png")

            if image_annotations is not None:
                image_annotations.root_dir = dataset.root_dir
                image_annotations["image_path"] = path

                # Plot transformed image
                fig = visualize.plot_annotations(
                    image=image,
                    annotations=image_annotations,
                    savedir=out_dir,
                    basename=basename,
                    thickness=self.thickness,
                    show=False,
                )
                plt.close(fig)
            else:
                # Save un-annotated image
                Image.fromarray(image).save(out_path)

            label_dict = getattr(self.pl_module, "label_dict", None)
            gt_layer = _df_to_comet_annotations(
                image_annotations, "ground truth", label_dict=label_dict
            )

            self._log_to_all(
                image=out_path,
                trainer=self.trainer,
                tag=f"{split} dataset sample",
                raw_image=image,
                annotation_layers=[gt_layer] if gt_layer else None,
            )

    def _log_last_predictions(self, trainer, pl_module):
        """Log sample of predictions + targets from last validation."""
        if self.prediction_samples == 0:
            return

        if len(pl_module.predictions) > 0:
            df = pd.concat(pl_module.predictions)
        else:
            df = pd.DataFrame()

        if df.empty or "image_path" not in df.columns:
            return

        out_dir = os.path.join(self.savedir, "predictions")
        os.makedirs(out_dir, exist_ok=True)

        dataset = trainer.val_dataloaders.dataset

        # Add root_dir to the dataframe
        if "root_dir" not in df.columns:
            df["root_dir"] = dataset.root_dir

        # Limit to n images, potentially randomly selected
        if self.select_random:
            selected_images = np.random.choice(
                df.image_path.unique(), self.prediction_samples
            )
        else:
            selected_images = df.image_path.unique()[: self.prediction_samples]

        # Ensure color is correctly assigned
        if self.color is None:
            num_classes = len(df["label"].unique())
            results_color = sv.ColorPalette.from_matplotlib("viridis", num_classes)
        else:
            results_color = self.color

        for image_name in selected_images:
            pred_df = df[df.image_path == image_name]

            targets = utilities.format_geometry(
                dataset.annotations_for_path(image_name, return_tensor=True), scores=False
            )

            # Assume that validation images are un-augmented
            basename = Path(image_name).stem + f"_{trainer.global_step}"
            fig = visualize.plot_results(
                basename=basename,
                results=pred_df,
                ground_truth=targets,
                savedir=out_dir,
                results_color=results_color,
                thickness=self.thickness,
                show=False,
            )
            plt.close(fig)

            # Pred metadata, if supported.
            stats = (
                pred_df["score"]
                .agg(
                    mean_confidence="mean",
                    max_confidence="max",
                    min_confidence="min",
                    std_confidence="std",
                )
                .to_dict()
            )

            metadata = {"pred_count": len(pred_df), "gt_count": len(targets)}
            metadata.update(stats)

            with open(os.path.join(out_dir, basename + ".json"), "w") as fp:
                json.dump(metadata, fp, indent=1)

            label_dict = getattr(pl_module, "label_dict", None)
            layers = []
            gt_layer = _df_to_comet_annotations(
                targets, "ground truth", label_dict=label_dict
            )
            if gt_layer is not None:
                layers.append(gt_layer)
            pred_layer = _df_to_comet_annotations(
                pred_df, "predictions", label_dict=label_dict
            )
            if pred_layer is not None:
                layers.append(pred_layer)

            try:
                raw_image = np.array(
                    Image.open(os.path.join(dataset.root_dir, image_name)).convert("RGB")
                )
            except Exception:
                raw_image = None

            self._log_to_all(
                image=os.path.join(out_dir, basename + ".png"),
                trainer=trainer,
                tag="prediction sample",
                metadata=metadata,
                raw_image=raw_image,
                annotation_layers=layers or None,
            )

    def _log_to_all(
        self,
        image: str,
        trainer,
        tag,
        metadata: dict | None = None,
        raw_image: np.ndarray | None = None,
        annotation_layers: list[dict] | None = None,
    ):
        """Log to all connected loggers.

        TensorBoard receives the pre-rendered PNG so the matplotlib overlay
        is visible inline. Comet receives the raw (unannotated) image with
        ``annotations`` so boxes / polygons render as interactive native
        overlays (see Comet's log_image annotations API).
        """
        try:
            img = np.array(Image.open(image).convert("RGB"))

            loggers = [lg for lg in trainer.loggers if hasattr(lg, "experiment")]

            tb = next((lg for lg in loggers if hasattr(lg.experiment, "add_image")), None)
            if tb is not None:
                tb.experiment.add_image(
                    tag=f"{tag}/{os.path.basename(image)}",
                    img_tensor=img,
                    global_step=trainer.global_step,
                    dataformats="HWC",
                )
                return

            comet = next(
                (lg for lg in loggers if hasattr(lg.experiment, "log_image")),
                None,
            )
            if comet is not None:
                meta = {
                    "image_name": os.path.basename(image),
                    "context": tag,
                    "step": trainer.global_step,
                }

                if metadata:
                    meta.update(metadata)

                # Native Comet annotations are in raw-image coordinates, so
                # drop them if we have to fall back to the rendered PNG (which
                # has matplotlib chrome and a different coord space).
                if raw_image is not None:
                    comet_image = raw_image
                    annotations = annotation_layers
                else:
                    comet_image = img
                    annotations = None

                comet.experiment.log_image(
                    comet_image,
                    name=tag,
                    step=trainer.global_step,
                    metadata=meta,
                    annotations=annotations,
                )

        except Exception as e:
            warnings.warn(f"Tried to log {image} exception raised: {e}", stacklevel=2)


class images_callback(ImagesCallback):
    def __init__(self, savedir, **kwargs):
        warnings.warn(
            "Please use ImagesCallback instead.", DeprecationWarning, stacklevel=2
        )
        super().__init__(save_dir=savedir, **kwargs)
