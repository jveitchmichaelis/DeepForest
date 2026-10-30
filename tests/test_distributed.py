"""Tests for gathering validation results to rank 0 across processes."""

import os
import socket

import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics.detection import MeanAveragePrecision

from deepforest import distributed
from deepforest.metrics import RecallPrecision

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_batch(rank: int) -> tuple[list[dict], list[dict], list[str]]:
    """Different images and box counts on each rank."""
    generator = torch.Generator().manual_seed(rank)
    preds, targets, names = [], [], []
    for i in range(rank + 2):
        n = i + 1
        xy = torch.rand(n, 2, generator=generator) * 300
        boxes = torch.cat([xy, xy + 20 + 10 * torch.rand(n, 2, generator=generator)], 1)
        targets.append({"boxes": boxes, "labels": torch.zeros(n, dtype=torch.int64)})
        preds.append(
            {
                "boxes": boxes + torch.rand(n, 4, generator=generator) * 8,
                "labels": torch.zeros(n, dtype=torch.int64),
                "scores": torch.rand(n, generator=generator),
            }
        )
        names.append(f"rank{rank}_image{i}.png")
    return preds, targets, names


def _metrics() -> tuple[MeanAveragePrecision, RecallPrecision]:
    return (
        MeanAveragePrecision(backend="faster_coco_eval", sync_on_compute=False),
        RecallPrecision(label_dict={"Tree": 0}, sync_on_compute=False),
    )


def _worker(rank: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        preds, targets, names = _make_batch(rank)
        map_metric, pr_metric = _metrics()
        map_metric.update(preds, targets)
        pr_metric.update(preds, targets, names)

        distributed.gather_metric(map_metric, dst=0)
        distributed.gather_metric(pr_metric, dst=0, extra_lists=("results",))

        frame = pd.DataFrame({"image_path": names})
        frames = distributed.gather_dataframe(frame, dst=0)

        result = None
        if rank == 0:
            result = {
                "map": {k: v for k, v in map_metric.compute().items() if k != "classes"},
                "pr": pr_metric.compute(),
                "results": len(pr_metric.get_results()),
                "frames": sorted(frames.image_path),
            }
        result = distributed.broadcast_object(result)
        queue.put((rank, result, len(frames)))
    finally:
        dist.destroy_process_group()


def test_gather_to_rank_zero_matches_single_process():
    """Metrics computed on rank 0 match one process that sees every image."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=_worker, args=(rank, port, queue))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    outputs = {}
    for _ in processes:
        rank, result, n_frames = queue.get(timeout=120)
        outputs[rank] = (result, n_frames)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    # Every rank receives rank 0's result, but only rank 0 holds the frames
    assert outputs[0][0] == outputs[1][0]
    assert outputs[1][1] == 0

    map_metric, pr_metric = _metrics()
    names = []
    for rank in range(WORLD_SIZE):
        preds, targets, rank_names = _make_batch(rank)
        map_metric.update(preds, targets)
        pr_metric.update(preds, targets, rank_names)
        names += rank_names
    expected_map = {k: v for k, v in map_metric.compute().items() if k != "classes"}
    expected_pr = pr_metric.compute()

    result = outputs[0][0]
    assert result["frames"] == sorted(names)
    assert result["results"] == len(pr_metric.get_results())
    for key, value in expected_map.items():
        assert result["map"][key] == pytest.approx(value.item()), key
    for key, value in expected_pr.items():
        assert result["pr"][key] == pytest.approx(value.item(), nan_ok=True), key
