"""Root-level config for sampling."""
from dataclasses import dataclass
from pathlib import Path

DATA_DIRECTORY = Path("/code/data")
DATASET_DIRECTORY = DATA_DIRECTORY / "datasets"


@dataclass
class Config:
    """Config values."""

    # Local paths
    data_directory: Path = Path("/code/data")
    dataset_directory: Path = data_directory / "datasets"
    models_directory: Path = data_directory / "models"
    images_directory: Path = dataset_directory / "images"
