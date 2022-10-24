"""Config for keypoint model."""
from dataclasses import dataclass
from pathlib import Path

from ..config import Config as _Config


@dataclass
class Config(_Config):
    """Config values."""

    # Local paths
    dataset_directory: Path = _Config.dataset_directory
    models_directory: Path = _Config.models_directory

    annotations_directory: Path = dataset_directory / "detection" / "annotations"
    predictions_directory: Path = dataset_directory / "detection" / "predictions"
    trained_model_path: Path = models_directory / "detection" / "model.pth"
    onnx_model_path: Path = models_directory / "detection" / "model.onnx"

    # Dataset
    min_size = 800

    # Model
    labels = ["vehicle"]
    n_classes: int = 1
    early_stopping_patience = 6
    trainable_backbone_layers: int = 2
    max_boxes: int = 20
    original_image_size: tuple[int, int] = (1280, 720)  # width, height

    # Training
    batch_size: int = 8
    n_workers: int = 4
    max_epochs: int = 100
    gpus: int = 1

    # HP Optimization
    learning_rate: float = 0.00025
