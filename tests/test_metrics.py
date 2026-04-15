import pandas as pd
import torch

from deepforest import get_data
from deepforest.metrics import RecallPrecision


def test_recall_precision_box():
    csv_path = get_data("example.csv")
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    # Convert label to int for metric compatibility
    label_map = {row["label"]: 1}
    pred = [
        {
            "boxes": torch.tensor(
                [[row.xmin + 1, row.ymin + 1, row.xmax, row.ymax]], dtype=torch.float32
            ),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
    ]
    target = [
        {
            "boxes": torch.tensor(
                [[row.xmin, row.ymin, row.xmax, row.ymax]], dtype=torch.float32
            ),
            "labels": torch.tensor([1]),
        }
    ]
    metric = RecallPrecision(task="box", label_dict=label_map)
    metric.update(pred, target)
    result = metric.compute()
    assert "box_precision" in result
    assert "box_recall" in result
    assert result["box_precision"] > 0
    assert result["box_recall"] > 0


def test_recall_precision_keypoint():
    csv_path = get_data("2019_BLAN_3_751000_4330000_image_crop_keypoints.csv")
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    label_map = {row["label"]: 1}
    pred = [
        {
            "points": torch.tensor([[row.x, row.y]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([row.score], dtype=torch.float32),
        }
    ]
    target = [
        {
            "points": torch.tensor([[row.x+1, row.y+1]], dtype=torch.float32),
            "labels": torch.tensor([1]),
        }
    ]
    metric = RecallPrecision(task="point", label_dict=label_map)
    metric.update(pred, target)
    result = metric.compute()
    assert "point_precision" in result
    assert "point_recall" in result
    assert result["point_precision"] > 0
    assert result["point_recall"] > 0
