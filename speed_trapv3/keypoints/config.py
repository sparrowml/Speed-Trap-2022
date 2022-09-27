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
    # darwin_team_slug: str = "hudl"
    # darwin_dataset_slug: str = "frame-annotations"

    # Model
    keypoint_names = ("back_tire",)
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    image_resize: tuple[int, int] = (480, 640)  # Height, width

    # TODO: we might want to make this paramete resolution independent
    # covariance_2d: float = 20

    # Training
    batch_size: int = 4
    num_workers: int = 4
    max_epochs: int = 100
    gpus: int = 1

    # HP Optimization
    learning_rate: float = 0.002247
