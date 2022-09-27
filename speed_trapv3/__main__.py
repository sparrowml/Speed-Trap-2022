"""Expose CLI."""
from typing import Callable

import fire

from .keypoints import commands as keypoints_commands

# from .tracking import commands as tracking_commands


def main():
    """Expose CLI."""
    fire.Fire(
        {
            "keypoints": keypoints_commands(),
            # "tracking": tracking_commands(),
        }
    )
