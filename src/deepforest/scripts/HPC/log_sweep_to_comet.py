"""Push evaluation sweep results into a single Comet experiment.

`deepforest evaluate -o` writes one `metric,value` CSV per run. This
collects a directory of them into one experiment, prefixing each metric
with the file name so the variants can be compared side by side.

    uv run python src/deepforest/scripts/HPC/log_sweep_to_comet.py \
        --sweep-dir logs/maskrcnn/<run>/<version>/checkpoints/sweep \
        --experiment-name oam-threshold-sweep --tag sweep
"""

import argparse
import csv
import os
from pathlib import Path


def read_metrics(path: Path) -> dict[str, float]:
    """Read a metric,value CSV written by ``deepforest evaluate -o``."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))

    return {row["metric"]: float(row["value"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True, help="Directory of result CSVs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--checkpoint", default=None, help="Logged as a parameter")
    parser.add_argument(
        "--offline", action="store_true", help="Write locally instead of uploading"
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    results = sorted(sweep_dir.glob("*.csv"))
    if not results:
        raise SystemExit(f"No CSVs in {sweep_dir}")

    import comet_ml

    kwargs = {"project_name": os.environ.get("COMET_PROJECT", "deepforest-maskrcnn")}
    if args.offline:
        experiment = comet_ml.OfflineExperiment(
            offline_directory=str(sweep_dir), **kwargs
        )
    else:
        experiment = comet_ml.Experiment(**kwargs)

    if args.experiment_name:
        experiment.set_name(args.experiment_name)
    if args.tag:
        experiment.add_tags(args.tag)
    if args.checkpoint:
        experiment.log_parameter("checkpoint", args.checkpoint)

    for path in results:
        metrics = read_metrics(path)
        experiment.log_metrics(metrics, prefix=path.stem)
        print(f"{path.stem}: {len(metrics)} metrics")

    experiment.end()


if __name__ == "__main__":
    main()
