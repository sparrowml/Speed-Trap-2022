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

    # JSON Paths
    json_directory: Path = dataset_directory / "tracking" / "predictions"

    # Rules threshold values
    rule0_x_min: int = 600
    rule0_x_max: int = 1100
    rule1_min_tire_dist: int = 20
    distance_error_threshold = 5
    INCLUSION_RADIUS = 20
    DETECTION_AREA_START_X = 800
    in_between_angle = 90 #in degrees

    # Calibration
    # REFERENCE: https://www.carwow.co.uk/guides/glossary/what-is-a-car-wheelbase-0282
    # WHEEL_BASE = 0.001509932  # IN MILES
    WHEEL_BASE = 2.43  # In meters
    MPERSTOMPH = 2.237 #METERS PER SECOND TO MILES PER HOUR
    
