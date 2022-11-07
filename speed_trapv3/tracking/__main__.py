"""Expose tracking CLI."""
from __future__ import annotations

from typing import Callable

from .add_keypoints import add_keypoints
from .tracking import track_objects, write_to_json

# from .visualize import visualize_tracking


def commands() -> dict[str, Callable[..., None]]:
    """Return tracking subcommands for CLI."""
    return {
        "track-objects": track_objects,
        "write-to-json": write_to_json,
        "add-keypoints": add_keypoints,
        # "visualize-tracking": visualize_tracking,
    }
