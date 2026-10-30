"""Helpers for distributed-safe logging and object gathering."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchmetrics import Metric


def is_distributed() -> bool:
    """Return True when torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def is_global_zero(trainer: Any | None = None) -> bool:
    """Return True on the global-zero rank."""
    if trainer is not None:
        return bool(getattr(trainer, "is_global_zero", True))

    if not is_distributed():
        return True

    return dist.get_rank() == 0


def should_sync(trainer: Any | None = None) -> bool:
    """Return True when metrics should be synchronized across ranks."""
    if trainer is not None:
        world_size = getattr(trainer, "world_size", 1)
        return world_size is not None and world_size > 1

    return is_distributed()


def get_rank() -> int:
    """Return the current distributed rank, defaulting to zero."""
    if not is_distributed():
        return 0

    return dist.get_rank()


def get_world_size() -> int:
    """Return the distributed world size, defaulting to one."""
    if not is_distributed():
        return 1

    return dist.get_world_size()


class FixedOrderSampler(Sampler[int]):
    """Yield a fixed sequence of dataset indices."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def gather_object(obj: Any, dst: int | None = None) -> list[Any] | None:
    """Gather a Python object from every rank.

    Args:
        obj: Object to send. It must be picklable.
        dst: Rank that receives the objects. If None, every rank receives them.

    Returns:
        The objects from every rank, in rank order. Ranks other than ``dst``
        get None.
    """
    if not is_distributed():
        return [obj]

    if dst is None:
        gathered = [None] * get_world_size()
        dist.all_gather_object(gathered, obj)
        return gathered

    gathered = [None] * get_world_size() if get_rank() == dst else None
    dist.gather_object(obj, gathered, dst=dst)
    return gathered


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Send a Python object from rank ``src`` to every rank."""
    if not is_distributed():
        return obj

    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def gather_dataframe(frame: pd.DataFrame, dst: int | None = None) -> pd.DataFrame:
    """Gather pandas DataFrames from every rank.

    Args:
        frame: DataFrame from this rank.
        dst: Rank that receives the frames. If None, every rank receives them
            and ranks other than ``dst`` get an empty frame.
    """
    gathered = gather_object(frame, dst=dst)
    if gathered is None:
        gathered = []
    non_empty_frames = [
        item for item in gathered if isinstance(item, pd.DataFrame) and not item.empty
    ]

    if non_empty_frames:
        return pd.concat(non_empty_frames, ignore_index=True)

    if isinstance(frame, pd.DataFrame):
        return frame.iloc[0:0].copy()

    return pd.DataFrame()


def gather_metric(
    metric: Metric, dst: int = 0, extra_lists: tuple[str, ...] = ()
) -> None:
    """Collect the states of a torchmetrics metric on rank ``dst``.

    Torchmetrics normally gathers the states to every rank, and every rank
    then computes the same result. This sends them to one rank only, so the
    other ranks do not hold a copy. Create the metric with
    ``sync_on_compute=False`` so ``compute()`` does not gather them again.
    Other ranks keep their local states.

    Args:
        metric: Metric to gather.
        dst: Rank that receives the states.
        extra_lists: Names of other list attributes to concatenate, for
            values that cannot be stored as metric states.
    """
    if not is_distributed():
        return

    names = list(metric.metric_state) + list(extra_lists)
    local = {name: _to_cpu(getattr(metric, name)) for name in names}
    gathered = gather_object(local, dst=dst)
    if gathered is None:
        return

    for name in names:
        values = [rank_values[name] for rank_values in gathered]
        if isinstance(values[0], list):
            merged = [item for rank_list in values for item in rank_list]
            merged = [_to_device(item, metric.device) for item in merged]
        else:
            stacked = torch.stack(values).to(metric.device)
            reduce_fn = metric._reductions[name]
            merged = reduce_fn(stacked) if reduce_fn is not None else stacked
        setattr(metric, name, merged)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value
