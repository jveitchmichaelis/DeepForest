import glob
import os
import subprocess
import sys
from importlib.resources import files

import pytest
from omegaconf import OmegaConf

from deepforest import get_data
from deepforest.main import deepforest

SCRIPT = files("deepforest.scripts").joinpath("cli.py")


def test_train_cli(tmpdir):
    """Check a basic training run, including overrides for unit testing
    see test_main.py fixtures for setup reference."""

    test_labels = get_data("OSBS_029.csv")

    args = [
        sys.executable,
        str(SCRIPT),
        "train",
        "train.fast_dev_run=True",
        f"train.csv_file={test_labels}",
        f"train.root_dir={os.path.dirname(test_labels)}"
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_train_cli_fail(tmpdir):
    """Check that training fails if no dataset paths are provided"""

    args = [
        sys.executable,
        str(SCRIPT),
        "train",
        "train.fast_dev_run=True",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode != 0


def test_train_cli_user_config(tmpdir):
    """Check whether we can provide a custom YAML file for configuration"""

    # Create a modified config
    test_labels = get_data("OSBS_029.csv")
    config = OmegaConf.load(get_data("config.yaml"))
    config.train.csv_file = test_labels
    config.train.root_dir = os.path.dirname(test_labels)
    OmegaConf.save(config, tmpdir.join("user_config.yaml").open('w'))

    # This will fail if the config is not correctly created
    # as the csv/root parameters are not set by default.
    args = [
        sys.executable,
        str(SCRIPT),
        f"--config-dir", tmpdir,
        f"--config-name", "user_config",
        "train",
        "train.fast_dev_run=True"
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_predict_cli(tmp_path):
    """Check we can predict an image and save results"""
    input_path = get_data("OSBS_029.png")
    output_path = tmp_path / "result.csv"
    args = [input_path, "-o", str(output_path)]

    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_predict_cli_with_opt(tmp_path):
    """Check we can predict an image and save results"""
    input_path = get_data("OSBS_029.png")
    output_path = tmp_path / "result.csv"
    args = [input_path, "-o", str(output_path), "patch_size=250"]

    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_predict_cli_missing_input(tmp_path):
    # Running the script without any inputs should yield an error
    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0


def test_predict_cli_config_help(tmp_path):
    # Script should show config without requiring input
    result = subprocess.run(
        [sys.executable, SCRIPT, "config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert len(result.stdout) > 0


@pytest.mark.parametrize("architecture", ["retinanet", "DeformableDetr"])
def test_train_pretrain_finetune_cli(tmpdir, architecture):
    """Test pretrain-then-finetune workflow via CLI using custom config files.

    This test validates:
    1. Training from scratch with model.name=None using a custom config
    2. Checkpoint saving during training
    3. Loading checkpoint via model.name for finetuning with a custom config
    4. Both retinanet and DeformableDetr architectures
    """

    # 1. Setup
    test_labels = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(test_labels)

    # 2. Pretrain phase with custom config
    pretrain_log_root = str(tmpdir / "pretrain_logs")

    # Create pretrain config
    pretrain_config = OmegaConf.load(get_data("config.yaml"))
    pretrain_config.architecture = architecture
    pretrain_config.model.name = None
    pretrain_config.train.csv_file = test_labels
    pretrain_config.train.root_dir = root_dir
    pretrain_config.train.epochs = 1
    pretrain_config.train.fast_dev_run = False
    pretrain_config.train.log_root = pretrain_log_root
    pretrain_config.validation.csv_file = test_labels
    pretrain_config.validation.root_dir = root_dir

    pretrain_config_path = tmpdir / "pretrain_config.yaml"
    OmegaConf.save(pretrain_config, pretrain_config_path.open('w'))

    pretrain_args = [
        sys.executable,
        str(SCRIPT),
        "--config-dir", str(tmpdir),
        "--config-name", "pretrain_config",
        "train",
    ]

    result = subprocess.run(
        pretrain_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"Pretrain failed:\nstderr: {result.stderr}\nstdout: {result.stdout}"

    # Find the checkpoint
    checkpoints = glob.glob(f"{pretrain_log_root}/**/checkpoints/last.ckpt", recursive=True)
    assert len(checkpoints) > 0, f"No checkpoint found in {pretrain_log_root}"
    pretrain_checkpoint = checkpoints[0]

    # Check we can load it
    m_pretrain = deepforest.load_from_checkpoint(
        pretrain_checkpoint,
        map_location="cpu",
        weights_only=True
    )
    assert m_pretrain.config.architecture == architecture
    del m_pretrain

    # 3. Finetune with custom config
    finetune_log_root = str(tmpdir / "finetune_logs")

    # Find the hf_weights directory (same pattern as checkpoint search)
    hf_weights_dirs = glob.glob(f"{pretrain_log_root}/**/hf_weights", recursive=True)
    assert len(hf_weights_dirs) > 0, f"No hf_weights directory found in {pretrain_log_root}"
    checkpoint_dir = hf_weights_dirs[0]

    # Create finetune config
    finetune_config = pretrain_config.copy()
    finetune_config.model.name = checkpoint_dir
    finetune_config.train.log_root = finetune_log_root

    finetune_config_path = tmpdir / "finetune_config.yaml"
    OmegaConf.save(finetune_config, finetune_config_path.open('w'))

    finetune_args = [
        sys.executable,
        str(SCRIPT),
        "--config-dir", str(tmpdir),
        "--config-name", "finetune_config",
        "train",
    ]

    result = subprocess.run(
        finetune_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"Finetune failed:\nstderr: {result.stderr}\nstdout: {result.stdout}"

    # Verify finetune checkpoint exists
    finetune_checkpoints = glob.glob(f"{finetune_log_root}/**/checkpoints/last.ckpt", recursive=True)
    assert len(finetune_checkpoints) > 0, f"No finetune checkpoint found in {finetune_log_root}"

    m_finetune = deepforest.load_from_checkpoint(
        finetune_checkpoints[0],
        map_location="cpu",
        weights_only=True
    )
    assert m_finetune.config.architecture == architecture
