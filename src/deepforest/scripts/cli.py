import argparse
import datetime
import glob
import os
import sys
import traceback
import warnings
from pathlib import Path

import torch
from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from deepforest.callbacks import EvaluationCallback, ImagesCallback
from deepforest.conf.schema import Config as StructuredConfig
from deepforest.main import deepforest
from deepforest.visualize import plot_results


def train(
    config: DictConfig,
    checkpoint: bool = True,
    comet: bool = False,
    tensorboard: bool = False,
    trace: bool = False,
    compress: bool = False,
    resume: str | None = None,
) -> bool:
    """Train a DeepForest model with configurable logging and experiment
    tracking.

    This training function sets up PyTorch Lightning trainer with various logging
    options including CSV, TensorBoard, and Comet ML. When Comet logging is enabled,
    the experiment ID is automatically captured and stored in the model's
    hyperparameters for later use.

    Training Workflows:
        1. Train from scratch: Set model.name=None in config (default)
        2. Fine-tune: Set model.name=<path_to_hf_weights_or_repo> in config to load pretrained
           weights and train on new data with fresh optimizer state
        3. Resume training: Use --resume <path_to_checkpoint.ckpt> to continue training
           from a checkpoint with optimizer/scheduler state intact

    After Training:
        - "Best" model is saved as HuggingFace weights in <log_dir>/hf_weights/
        - Best and last checkpoints are saved to <log_dir>/checkpoints/*.ckpt

    Args:
        config (DictConfig): Hydra configuration containing model and training parameters
        checkpoint (bool, optional): Whether to enable model checkpointing. Defaults to True.
        comet (bool, optional): Whether to enable Comet ML logging. Requires comet-ml
            package and proper environment variables (COMET_API_KEY, COMET_WORKSPACE).
            Defaults to False.
        tensorboard (bool, optional): Whether to enable TensorBoard logging in addition
            to CSV logging. Defaults to False.
        trace (bool, optional): Whether to enable PyTorch memory profiling for debugging.
            Only works when CUDA is available. Defaults to False.
        compress (bool, optional): Whether to compress prediction CSV files using gzip for
            better storage efficiency. Defaults to False.
        resume (str, optional): Path to checkpoint file to resume training from. Preserves
            optimizer state and training progress. For fine-tuning with new data, use
            model.name in config instead. Defaults to None.

    Returns:
        bool: True if training completed successfully, False if training failed

    Note:
        When Comet logging is enabled, the experiment ID (key) is automatically added
        to the model's hyperparameters as 'experiment_id' for later re-logging to
        the same experiment.
    """

    if trace:
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available, skipping trace.", stacklevel=2)
        else:
            torch.cuda.memory._record_memory_history()

    # Create model - either from scratch or with HF pretrained weights via config.model.name
    m = deepforest(config=config)

    # Validate resume checkpoint if provided
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        m.print(f"Resuming training from checkpoint: {resume}")

    # Warn if model.name looks like a checkpoint path. In this case the
    # user should provide a path to either a repo, or a pretrained checkpoint
    # directory (safetensors + config)
    if (
        config.model.name is not None
        and os.path.exists(config.model.name)
        and config.model.name.endswith(".ckpt")
    ):
        warnings.warn(
            f"model.name points to a .ckpt file: {config.model.name}\n"
            "For resuming training, use --resume flag instead.\n"
            "model.name should point to HuggingFace weights directory.",
            stacklevel=2,
        )

    # Set matmul precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    callbacks = []
    loggers = []
    log_root = Path(config.train.log_root)

    # Use defaults from Lightning unless overriden by Comet
    experiment_name = None
    experiment_id = None
    # Store as %YYYY%mm%ddT%HH:%MM:%SS
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract experiment_id for Comet experiment continuation
    if resume is not None and os.path.exists(resume):
        # Resuming from a checkpoint - need to peek at the checkpoint file
        # since the model may have been loaded from a different source
        try:
            resume_ckpt = torch.load(resume, map_location="cpu", weights_only=False)
            if (
                "hyper_parameters" in resume_ckpt
                and "experiment_id" in resume_ckpt["hyper_parameters"]
            ):
                experiment_id = resume_ckpt["hyper_parameters"]["experiment_id"]
                print(f"Resuming training with experiment_id: {experiment_id}")
        except Exception as e:
            warnings.warn(
                f"Failed to extract experiment_id from resume checkpoint: {e}",
                stacklevel=2,
            )

    if comet and not m.config.train.fast_dev_run:
        try:
            # Comet setup requires an external dependency
            from pytorch_lightning.loggers import CometLogger

            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                project=os.environ.get("COMET_PROJECT", default="DeepForest"),
                experiment_key=experiment_id,
                offline_directory=config.train.log_root,
            )

            # Get experiment_id if we aren't resuming
            if experiment_id is None:
                experiment_id = comet_logger.experiment.get_key()

            # Comet will set the experiment name for us
            experiment_name = comet_logger.experiment.get_name()
            version = ""
            loggers.append(comet_logger)
        except ImportError:
            warnings.warn(
                "Failed to import Comet, check if comet-ml is installed", stacklevel=2
            )
        except Exception as e:
            warnings.warn(f"Failed to set up comet logger. {e}", stacklevel=2)
    else:
        callbacks.append(DeviceStatsMonitor())

    # By default, create a CSV logger and monitor stats
    csv_logger = CSVLogger(save_dir=log_root, name=experiment_name, version=version)
    loggers.append(csv_logger)

    if tensorboard:
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            sub_dir="tensorboard",
            name=experiment_name,
            version=version,
        )
        loggers.append(tensorboard_logger)

    callbacks.append(
        ImagesCallback(
            save_dir=Path(csv_logger.log_dir) / "images",
            every_n_epochs=config.validation.val_accuracy_interval,
            select_random=True,
        )
    )

    evaluation_path = Path(csv_logger.log_dir) / "predictions"
    callbacks.append(
        EvaluationCallback(
            save_dir=evaluation_path,
            compress=compress,
            every_n_epochs=config.validation.val_accuracy_interval,
            run_evaluation=True,
        )
    )

    # Setup checkpoint to store in log directory
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(csv_logger.log_dir) / "checkpoints",
            filename=f"{config.architecture}-{{epoch:02d}}-{{map_50:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        # Using equals causes a lot of strife with Hydra, so use colon instead.
        checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
        callbacks.append(checkpoint_callback)

    m.create_trainer(
        logger=loggers,
        callbacks=callbacks,
        strategy="ddp_find_unused_parameters_true"
        if torch.cuda.is_available()
        and config.model.name is not None
        and "dino" in config.model.name.lower()
        else "auto",
    )
    # Add experiment ID to hyperparameters if available
    if experiment_id is not None:
        # Update the saved hyperparameters to include experiment ID
        current_hparams = m.hparams.copy()
        current_hparams["experiment_id"] = experiment_id
        m.save_hyperparameters(current_hparams)

    train_success = False
    try:
        m.trainer.fit(m, ckpt_path=resume, weights_only=False)
        train_success = True
    except Exception as e:
        warnings.warn(
            f"Training failed with exception {e}. Will attempt to upload any existing checkpoints if enabled.",
            stacklevel=2,
        )
        warnings.warn(traceback.format_exc(), stacklevel=2)

    if trace and torch.cuda.is_available():
        torch.cuda.memory._dump_snapshot(
            filename=Path(csv_logger.log_dir) / "dump_snapshot.pickle"
        )

    # Save checkpoints
    if train_success and checkpoint:
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint and os.path.exists(best_checkpoint):
            m.print(f"Saving HuggingFace weights from best checkpoint: {best_checkpoint}")
            hf_weights_dir = Path(csv_logger.log_dir) / "hf_weights"

            # Load best checkpoint and save as HF weights
            best_model = deepforest.load_from_checkpoint(
                best_checkpoint, map_location="cpu", weights_only=True
            )
            best_model.model.save_pretrained(hf_weights_dir)
            m.print(f"HuggingFace weights saved to: {hf_weights_dir}")

            # Print summary of saved artifacts
            m.print("\nTraining artifacts saved:")
            m.print(f"  - Best model (HF weights): {hf_weights_dir}")
            m.print(f"  - Best checkpoint (.ckpt): {best_checkpoint}")
            m.print(
                f"  - Last checkpoint (.ckpt): {Path(checkpoint_callback.dirpath) / 'last.ckpt'}"
            )
            m.print("\nFor fine-tuning: use model.name={hf_weights_dir}")
            m.print(
                f"For resuming: use --resume {Path(checkpoint_callback.dirpath) / 'last.ckpt'}"
            )

            # Log to comet if enabled
            if comet:
                for logger in m.trainer.loggers:
                    if hasattr(logger.experiment, "log_artifact"):
                        logger.experiment.log_asset_folder(
                            hf_weights_dir, log_file_name=True
                        )

    # Upload predictions
    if comet:
        for logger in m.trainer.loggers:
            m.print("Uploading predictions")

            if hasattr(logger.experiment, "log_artifact"):
                logger.experiment.log_asset_folder(evaluation_path, log_file_name=True)

    if checkpoint:
        for logger in m.trainer.loggers:
            if hasattr(logger.experiment, "log_model"):
                for checkpoint in glob.glob(
                    os.path.join((checkpoint_callback.dirpath), "*.ckpt")
                ):
                    m.print(f"Uploading checkpoint {checkpoint}")
                    logger.experiment.log_model(
                        name=os.path.basename(checkpoint), file_or_folder=str(checkpoint)
                    )

    return train_success


def predict(
    config: DictConfig,
    input_path: str | None = None,
    output_path: str | None = None,
    plot: bool | None = False,
    root_dir: str | None = None,
) -> None:
    """Run prediction for the given image or CSV file, optionally saving the
    results to the provided path and optionally visualizing the results.

    Args:
        config (DictConfig): Hydra configuration.
        input_path (Optional[str]): Path to the input image or CSV file. If None, uses config.validation.csv_file.
        output_path (Optional[str]): Path to save the prediction results.
        plot (Optional[bool]): Whether to plot the results.
        root_dir (Optional[str]): Root directory containing images when input_path is a CSV file.

    Returns:
        None
    """

    # For inference, check if model.name is a PyTorch Lightning checkpoint
    if config.model.name is not None and config.model.name.endswith(".ckpt"):
        # Load from checkpoint for inference
        map_location = "cpu" if config.accelerator == "auto" else config.accelerator
        m = deepforest.load_from_checkpoint(
            config.model.name, map_location=map_location, weights_only=True
        )
    else:
        # Load from HF weights or create new model
        m = deepforest(config=config)

    m.create_trainer(logger=False)

    # Use validation CSV from config if not provided
    if input_path is None:
        if config.validation.csv_file is None:
            raise ValueError(
                "No input file provided and config.validation.csv_file is not set"
            )
        input_path = config.validation.csv_file
        print(f"Using validation CSV from config: {input_path}")

    # Use validation root_dir from config if not provided and input is CSV
    if input_path.endswith(".csv") and root_dir is None:
        root_dir = config.validation.root_dir
        if root_dir is not None:
            print(f"Using root directory from config: {root_dir}")

    if input_path.endswith(".csv"):
        # CSV batch prediction
        res = m.predict_file(
            csv_file=input_path,
            root_dir=root_dir,
            batch_size=config.batch_size,
        )
    else:
        # Single image prediction
        res = m.predict_tile(
            path=input_path,
            patch_size=config.patch_size,
            patch_overlap=config.patch_overlap,
            iou_threshold=config.nms_thresh,
        )

    if output_path is not None:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        plot_results(res)


def evaluate(
    config: DictConfig,
    csv_file: str | None = None,
    root_dir: str | None = None,
    predictions_csv: str | None = None,
    iou_threshold: float | None = None,
    batch_size: int | None = None,
    size: int | None = None,
    experiment_id: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run evaluation on ground truth annotations, optionally logging to Comet.

    Args:
        config (DictConfig): Hydra configuration.
        csv_file (Optional[str]): Path to ground truth CSV file with annotations. If None, uses config.validation.csv_file.
        root_dir (Optional[str]): Root directory containing images. If None, uses config value or directory of csv_file.
        predictions_csv (Optional[str]): Path to predictions CSV file. If None, generates predictions.
        iou_threshold (Optional[float]): IoU threshold for evaluation. If None, uses config value.
        batch_size (Optional[int]): Batch size for prediction. If None, uses config value.
        size (Optional[int]): Size to resize images for prediction. If None, no resizing.
        experiment_id (Optional[str]): Comet experiment ID to log results to.
        output_path (Optional[str]): Path to save evaluation results CSV.

    Returns:
        None
    """
    # For inference, check if model.name is a PyTorch Lightning checkpoint
    if config.model.name is not None and config.model.name.endswith(".ckpt"):
        # Load from checkpoint for inference
        map_location = "cpu" if config.accelerator == "auto" else config.accelerator
        m = deepforest.load_from_checkpoint(
            config.model.name, map_location=map_location, weights_only=True
        )
    else:
        # Load from HF weights or create new model
        m = deepforest(config=config)

    m.create_trainer(logger=False)

    # Use validation CSV from config if not provided
    if csv_file is None:
        if config.validation.csv_file is None:
            raise ValueError(
                "No CSV file provided and config.validation.csv_file is not set"
            )
        csv_file = config.validation.csv_file
        print(f"Using validation CSV from config: {csv_file}")

    # Use validation root_dir from config if not provided
    if root_dir is None:
        root_dir = config.validation.root_dir
        if root_dir is not None:
            print(f"Using root directory from config: {root_dir}")

    # Run evaluation
    results = m.evaluate(
        csv_file=csv_file,
        root_dir=root_dir,
        iou_threshold=iou_threshold,
        batch_size=batch_size,
        size=size,
        predictions=predictions_csv,
    )

    # Print results to console
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in results.items():
        if key not in ["predictions", "results", "ground_df", "class_recall"]:
            if value is not None:
                print(f"{key}: {value}")

    # Print class-specific results if available
    if results.get("class_recall") is not None:
        print("\nClass-specific Results:")
        print("-" * 30)
        for _, row in results["class_recall"].iterrows():
            label_name = m.numeric_to_label_dict[row["label"]]
            print(
                f"{label_name} - Recall: {row['recall']:.4f}, Precision: {row['precision']:.4f}"
            )

    # Log to Comet if experiment ID provided
    if experiment_id is not None:
        try:
            from pytorch_lightning.loggers import CometLogger

            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                project=os.environ.get("COMET_PROJECT", default="DeepForest"),
                experiment_key=experiment_id,  # Re-log to existing experiment
            )

            # Log evaluation metrics
            for key, value in results.items():
                if key not in ["predictions", "results", "ground_df", "class_recall"]:
                    if value is not None:
                        comet_logger.experiment.log_metric(key, value)

            # Log class-specific metrics
            if results.get("class_recall") is not None:
                for _, row in results["class_recall"].iterrows():
                    label_name = m.numeric_to_label_dict[row["label"]]
                    comet_logger.experiment.log_metric(
                        f"{label_name}_Recall", row["recall"]
                    )
                    comet_logger.experiment.log_metric(
                        f"{label_name}_Precision", row["precision"]
                    )

            print(f"\nResults logged to Comet experiment: {experiment_id}")

        except ImportError:
            warnings.warn(
                "Failed to import Comet, skipping experiment logging", stacklevel=2
            )
        except Exception as e:
            warnings.warn(f"Failed to log to Comet experiment. {e}", stacklevel=2)

    # Save results to CSV if output path provided
    if output_path is not None:
        import pandas as pd

        # Create a summary dataframe with evaluation metrics
        summary_data = []
        for key, value in results.items():
            if key not in ["predictions", "results", "ground_df", "class_recall"]:
                if value is not None:
                    summary_data.append({"metric": key, "value": value})

        # Add class-specific results if available
        if results.get("class_recall") is not None:
            for _, row in results["class_recall"].iterrows():
                label_name = m.numeric_to_label_dict[row["label"]]
                summary_data.append(
                    {"metric": f"{label_name}_Recall", "value": row["recall"]}
                )
                summary_data.append(
                    {"metric": f"{label_name}_Precision", "value": row["precision"]}
                )

        summary_df = pd.DataFrame(summary_data)
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"\nEvaluation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DeepForest CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model. Use --resume to continue training, or set model.name in config to fine-tune from pretrained weights.",
        epilog="""
Examples:
  # Train from scratch
  deepforest train

  # Fine-tune from HuggingFace weights
  deepforest train model.name=/path/to/hf_weights

  # Resume interrupted training
  deepforest train --resume /path/to/checkpoint.ckpt

Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.
        """,
    )
    train_parser.add_argument(
        "--disable-checkpoint", help="Path to log folder", action="store_true"
    )
    train_parser.add_argument(
        "--comet",
        help="Enable logging to Comet ML, requires comet to be logged in.",
        action="store_true",
    )
    train_parser.add_argument(
        "--tensorboard",
        help="Enable logging to Tensorboard",
        action="store_true",
    )
    train_parser.add_argument(
        "--trace",
        help="Enable PyTorch memory profiling.",
        action="store_true",
    )
    train_parser.add_argument(
        "--compress",
        help="Compress prediction CSV files using gzip for better storage efficiency.",
        action="store_true",
    )
    train_parser.add_argument(
        "--resume",
        help="Resume training from a PyTorch Lightning checkpoint (.ckpt file). "
        "For fine-tuning with new data, use model.name in config instead.",
        type=str,
        default=None,
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input image or CSV file",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    predict_parser.add_argument(
        "input",
        nargs="?",
        help="Path to input image or CSV file (optional if validation CSV specified in config)",
    )
    predict_parser.add_argument("-o", "--output", help="Path to prediction results")
    predict_parser.add_argument("--plot", action="store_true", help="Plot results")
    predict_parser.add_argument(
        "--root-dir", help="Root directory containing images (required when input is CSV)"
    )

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation on ground truth annotations",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    evaluate_parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to ground truth CSV file (optional if specified in config)",
    )
    evaluate_parser.add_argument("--root-dir", help="Root directory containing images")
    evaluate_parser.add_argument(
        "--predictions-csv",
        help="Path to predictions CSV file (if not provided, predictions will be generated)",
    )
    evaluate_parser.add_argument(
        "--iou-threshold", type=float, help="IoU threshold for evaluation"
    )
    evaluate_parser.add_argument(
        "--batch-size", type=int, help="Batch size for prediction"
    )
    evaluate_parser.add_argument(
        "--size", type=int, help="Size to resize images for prediction"
    )
    evaluate_parser.add_argument(
        "--experiment-id", help="Comet experiment ID to log results to"
    )
    evaluate_parser.add_argument(
        "-o", "--output", help="Path to save evaluation results CSV"
    )

    # Show config subcommand
    subparsers.add_parser("config", help="Show the current config")

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Show available config overrides and exit")
    parser.add_argument(
        "--config-name", help="Show available config overrides and exit", default="config"
    )

    args, overrides = parser.parse_known_args()

    if args.config_dir is not None:
        initialize_config_dir(version_base=None, config_dir=args.config_dir)
    else:
        initialize(version_base=None, config_path="pkg://deepforest.conf")

    base = OmegaConf.structured(StructuredConfig)
    cfg = compose(config_name=args.config_name, overrides=overrides)
    cfg = OmegaConf.merge(base, cfg)

    if args.command == "predict":
        predict(
            cfg,
            input_path=args.input,
            output_path=args.output,
            plot=args.plot,
            root_dir=args.root_dir,
        )
    elif args.command == "train":
        res = train(
            cfg,
            checkpoint=not args.disable_checkpoint,
            comet=args.comet,
            tensorboard=args.tensorboard,
            trace=args.trace,
            compress=args.compress,
            resume=args.resume,
        )

        sys.exit(0 if res else 1)

    elif args.command == "evaluate":
        evaluate(
            cfg,
            csv_file=args.csv_file,
            root_dir=args.root_dir,
            predictions_csv=args.predictions_csv,
            iou_threshold=args.iou_threshold,
            batch_size=args.batch_size,
            size=args.size,
            experiment_id=args.experiment_id,
            output_path=args.output,
        )

    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
