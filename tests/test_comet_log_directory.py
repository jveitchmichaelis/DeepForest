"""Comet run names identify local training output directories."""

from deepforest.scripts.train import train
from deepforest.utilities import load_config


def test_comet_run_directory_has_no_timestamp(tmp_path, mocker):
    config = load_config(overrides={"log_root": str(tmp_path)})
    model = mocker.patch("deepforest.scripts.train.deepforest").return_value
    model.config = config
    mocker.patch("deepforest.scripts.train.distributed.is_global_zero", return_value=True)
    logger = mocker.patch("pytorch_lightning.loggers.CometLogger").return_value
    logger.experiment.get_name.return_value = "steady_archway_4101"
    logger.experiment.get_key.return_value = "example-key"

    assert train(config, comet=True, checkpoint=False)

    run_dir = tmp_path / "steady_archway_4101"
    assert (run_dir / "config.yaml").is_file()
    assert not any(path.is_dir() for path in run_dir.iterdir())
