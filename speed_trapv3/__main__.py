"""Expose CLI."""
from typing import Callable

import fire

from .detection import commands as detection_commands
from .keypoints import commands as keypoints_commands
from .speed_estimation import commands as speed_estimation_commands
from .tracking import commands as tracking_commands


def main():
    """Expose CLI."""
    fire.Fire(
        {
            "keypoints": keypoints_commands(),
            "detection": detection_commands(),
            "tracking": tracking_commands(),
            "speed": speed_estimation_commands(),
        }
    )
