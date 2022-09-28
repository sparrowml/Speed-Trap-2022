"""Expose keypoints CLI."""
from typing import Callable

from speed_trapv3.keypoints.dataset import version_annotations

from .train import save_checkpoint, train_model


def commands() -> dict[str, Callable[..., None]]:
    """Return keypoints subcommands for CLI."""
    return {
        "version-annotations": version_annotations,
        "save-checkpoint": save_checkpoint,
        "train-model": train_model,
    }
