"""Expose keypoints CLI."""
from typing import Callable

from speed_trapv3.keypoints.datasets import version_annotations


def commands() -> dict[str, Callable[..., None]]:
    """Return keypoints subcommands for CLI."""
    return {"version-annotations": version_annotations}
