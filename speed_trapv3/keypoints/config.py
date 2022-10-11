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

    annotations_directory: Path = dataset_directory / "annotations"
    predictions_directory: Path = dataset_directory / "predictions"
    trained_model_path: Path = models_directory / "model.pth"

    # Dataset
    absent_class_pad_values: tuple[float, float] = (
        -1.0,
        -1.0,
    )  # any negative value would work!

    # Model
    keypoint_names = ("back_tire", "front_tire")
    original_image_size: tuple[int, int] = (1280, 720)  # width, height
    image_crop_size: tuple[int, int] = (
        762,  # width
        387,  # height
    )  # This value was derived by finding the maximum bounding box size for vehicles from the annotation files.
    num_classes: int = 2
    patience = 5

    # TODO: we might want to make this paramete resolution independent
    covariance_2d: float = 20

    # Training
    batch_size: int = 4
    num_workers: int = 4
    max_epochs: int = 100
    gpus: int = 1

    # HP Optimization
    learning_rate: float = 0.002247
