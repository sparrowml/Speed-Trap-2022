"""Utils for keypoint detection."""
import enum
from pathlib import Path
from typing import Optional, Union

import farmhash
import numpy as np
import numpy.typing as npt
import torch
from sparrow_datums import FrameAugmentedBoxes, PType
from sparrow_tracky import MODA, compute_moda_by_class

from .config import Config


class Holdout(enum.Enum):
    """Holdout enum."""

    TRAIN = 1
    DEV = 2
    TEST = 3


def get_holdout(label: str) -> Holdout:
    """Map a class hash to the corresponding holdout enum."""
    class_hash = farmhash.fingerprint64(label) % 10
    if class_hash < 8:
        return Holdout.TRAIN
    elif class_hash == 8:
        return Holdout.DEV
    else:
        return Holdout.TEST


def to_numpy(x: torch.Tensor) -> npt.NDArray[np.float64]:
    """Convert torch.Tensor to np.ndarray."""
    result: npt.NDArray[np.float64] = x.detach().cpu().numpy()
    return result


def result_to_boxes(
    result: dict[str, torch.Tensor],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> FrameAugmentedBoxes:
    """Convert retinanet result dict to FrameAugmentedBoxes."""
    box_data = to_numpy(result["boxes"]).astype("float64")
    labels = to_numpy(result["labels"]).astype("float64")
    if "scores" in result:
        scores = to_numpy(result["scores"]).astype("float64")
    else:
        scores = np.ones(len(labels))
    mask = scores >= 0
    box_data = box_data[mask]
    labels = labels[mask]
    scores = scores[mask]
    return FrameAugmentedBoxes(
        np.concatenate([box_data, scores[:, None], labels[:, None]], -1),
        ptype=PType.absolute_tlbr,
        image_width=image_width,
        image_height=image_height,
    )


def batch_moda(
    results: list[dict[str, torch.Tensor]],
    batch: list[dict[str, torch.Tensor]],
    score_threshold: float = 0.5,
) -> MODA:
    """Compute MODA for a batch of data."""
    moda = MODA()
    for result, sample in zip(results, batch):
        predicted_boxes = result_to_boxes(result)
        predicted_boxes = predicted_boxes[predicted_boxes.scores > score_threshold]
        ground_truth_boxes = result_to_boxes(sample)
        moda_dict = compute_moda_by_class(predicted_boxes, ground_truth_boxes)
        moda += sum(moda_dict.values())
    return moda


def get_annotation_path(slug: str) -> Path:
    """Return the annotation path for a slug."""
    return Config.annotations_directory / f"{slug}.json.gz"


def slugify(file_path: Union[str, Path]) -> str:
    """Get the slug from a filepath."""
    filename = Path(file_path).name
    filename = filename.removesuffix(".json.gz")
    filename = filename.removesuffix(".jpg")
    return filename


def get_prediction_path(slug: str) -> Path:
    """Return the prediction path for a slug."""
    return Config.predictions_directory / f"{slug}.json.gz"


def get_image_path(slug: str) -> Path:
    """Return the image path for a slug."""
    return Config.images_directory / f"{slug}.jpg"
