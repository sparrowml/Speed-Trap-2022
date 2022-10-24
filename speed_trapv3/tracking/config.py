"""Config for tracking."""
from dataclasses import dataclass
from pathlib import Path

from ..config import Config as _Config


@dataclass
class Config(_Config):
    """Config values."""

    image_width: int = 1280
    image_height: int = 720
    models_directory: Path = _Config.models_directory / "detection"
    dataset_directory: Path = _Config.dataset_directory / "tracking"
    annotations_directory: Path = dataset_directory / "annotations"
    onnx_model_path: Path = models_directory / "model.onnx"
    prediction_directory: Path = dataset_directory / "predictions"

    # Tracking hyperparameters
    vehicle_class_index: int = 0
    vehicle_euclidean_threshold: float = 0.5
    vehicle_score_threshold: float = 0.3
    vehicle_tracklet_length: int = 3
    vehicle_iou_threshold: float = 0.5

    # Visualization
    labels = "vehicle"
    colors = "tab:magenta"

    # Resampling
    resampled_fps: int = 10
