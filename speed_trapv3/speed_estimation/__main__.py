"""Expose tracking CLI."""
from __future__ import annotations

from typing import Callable

from .estimate_speed import estimate_speed
from .results import write_results


def commands() -> dict[str, Callable[..., None]]:
    """Return tracking subcommands for CLI."""
    return {
        "estimate-speed": estimate_speed,
        "write-results": write_results,
    }
