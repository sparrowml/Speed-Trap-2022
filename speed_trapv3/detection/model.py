import os
import tempfile
from pathlib import Path
from typing import Optional, Union, overload

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
from PIL import Image
from sparrow_datums import FrameAugmentedBoxes
from torchvision import models

from ..utils import result_to_boxes
from .config import Config


class RetinaNet(torch.nn.Module):
    """Retinanet model based on Torchvision"""

    def __init__(
        self,
        n_classes: int = Config.n_classes,
        min_size: int = Config.min_size,
        trainable_backbone_layers: int = Config.trainable_backbone_layers,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.model = models.detection.retinanet_resnet50_fpn(
            progress=True,
            pretrained=pretrained,
            num_classes=n_classes,
            min_size=min_size,
            trainable_backbone_layers=trainable_backbone_layers,
            pretrained_backbone=pretrained_backbone,
        )

    @overload
    def forward(self, x: npt.NDArray[np.float64]) -> FrameAugmentedBoxes:
        ...

    @overload
    def forward(self, x: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        ...

    @overload
    def forward(
        self, x: list[torch.Tensor], y: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        ...

    def forward(
        self,
        x: Union[npt.NDArray[np.float64], list[torch.Tensor]],
        y: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> Union[
        dict[str, torch.Tensor], list[dict[str, torch.Tensor]], FrameAugmentedBoxes
    ]:
        """
        Forward pass for training and inference.
        Parameters
        ----------
        x
            A list of image tensors with shape (3, n_rows, n_cols) with
            unnormalized values in [0, 1].
        y
            An optional list of targets with an x1, x2, y1, y2 "boxes" tensor
            and a class index "labels" tensor.
        Returns
        -------
        result(s)
            If inference, this will be a list of dicts with predicted tensors
            for "boxes", "scores" and "labels" in each one. If training, this
            will be a dict with loss tensors for "classification" and
            "bbox_regression".
        """
        if isinstance(x, np.ndarray):
            return self.forward_numpy(x)
        elif self.training:
            return self.model.forward(x, y)
        results = self.model.forward(x, y)
        padded_results = []
        for result in results:
            padded_result: dict[str, torch.Tensor] = {}
            padded_result["boxes"] = torch.nn.functional.pad(
                result["boxes"], (0, 0, 0, Config.max_boxes), value=-1.0
            )[: Config.max_boxes]
            padded_result["scores"] = torch.nn.functional.pad(
                result["scores"], (0, Config.max_boxes), value=-1.0
            )[: Config.max_boxes]
            padded_result["labels"] = torch.nn.functional.pad(
                result["labels"], (0, Config.max_boxes), value=-1
            )[: Config.max_boxes].float()
            padded_results.append(padded_result)
        return padded_results

    def forward_numpy(self, img: npt.NDArray[np.float64]) -> FrameAugmentedBoxes:
        """Run inference for np.ndarray."""
        image_height, image_width = img.shape[:2]
        x: torch.Tensor = T.ToTensor()(Image.fromarray(img))
        if torch.cuda.is_available():
            x = x.cuda()
        result = self.forward([x])[0]
        return result_to_boxes(
            result,
            image_width=image_width,
            image_height=image_height,
        )

    def load(self, model_path: str, skip_classes: bool = False) -> None:
        """Load model weights."""
        weights = torch.load(model_path)
        if skip_classes:
            del weights["model.head.classification_head.cls_logits.weight"]
            del weights["model.head.classification_head.cls_logits.bias"]
        strict = not skip_classes
        self.load_state_dict(weights, strict=strict)


def save_pretrained() -> None:
    """Save pretrained model."""
    model = RetinaNet(n_classes=91, pretrained=True)
    torch.save(model.state_dict(), Config.pretrained_model_path)
    print("Version pretrained model:")
    print(f"dvc add {Config.pretrained_model_path}")


def export_model(
    input_shape: tuple[int, int, int] = (
        3,
        Config.original_image_size[1],
        Config.original_image_size[0],
    )
) -> None:
    """Export ONNX model."""
    x = torch.randn(*input_shape)
    model = RetinaNet().eval()
    model.load(Config.trained_model_path)
    print(f"Input shape: {input_shape}")
    torch.onnx.export(
        model,
        [x],
        Config.onnx_model_path,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        opset_version=11,
    )
    print("Version ONNX model:")
    print(f"dvc add {Config.onnx_model_path}")
