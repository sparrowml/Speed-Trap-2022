"""Keypoint detection model architecture."""
from pathlib import Path
from typing import Union

import torch
from torchvision.models.segmentation import fcn_resnet50


class SegmentationModel(torch.nn.Module):
    """Model for prediction court/net key points."""

    def __init__(self) -> None:
        super().__init__()
        self.fcn = fcn_resnet50(num_classes=2, pretrained_backbone=True, aux_loss=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with the model."""
        heatmaps = torch.sigmoid(self.fcn(x)["out"])
        ncols = heatmaps.shape[-1]
        flattened_keypoint_indices = torch.flatten(heatmaps, 2).argmax(-1)
        xs = (flattened_keypoint_indices % ncols).float()
        ys = torch.floor(flattened_keypoint_indices / ncols)
        keypoints = torch.stack([xs, ys], -1)
        return {"heatmaps": heatmaps, "keypoints": keypoints}

    def load(self, model_path: Union[Path, str]) -> str:
        """Load model weights."""
        weights = torch.load(model_path)
        result: str = self.load_state_dict(weights)
        return result
