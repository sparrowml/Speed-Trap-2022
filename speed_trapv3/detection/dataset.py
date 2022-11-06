import enum
import json
import os
from operator import itemgetter
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from farmhash import hash64withseed
from PIL import Image
from sparrow_datums import FrameAugmentedBoxes, PType
from tqdm import tqdm

from ..utils import Holdout, get_holdout
from .config import Config


def version_detection_annotations(darwin_annotations_directory: str) -> None:
    """Convert V7 annotations to sparrow-datums format."""
    for darwin_path in Path(darwin_annotations_directory).glob("*.json"):
        boxes = FrameAugmentedBoxes.from_darwin_file(darwin_path, Config.labels)
        slug, _ = os.path.splitext(darwin_path.name)
        annotation_filename = f"{slug}.json.gz"
        annotation_path = Config.annotations_directory / annotation_filename
        boxes.to_file(annotation_path)


def get_sample_dicts(holdout: Optional[Holdout] = None) -> list[dict[str, Any]]:
    """Get sample dicts."""
    slugs = set(
        [
            p.name.removesuffix(".json.gz")
            for p in Config.annotations_directory.glob("*.json.gz")
        ]
    )
    samples = []

    for slug in slugs:
        sample_holdout = get_holdout(slug)
        if holdout and holdout != sample_holdout:
            continue
        image_path = Config.images_directory / f"{slug}.jpg"
        annotation_path = Config.annotations_directory / f"{slug}.json.gz"
        boxes = FrameAugmentedBoxes.from_file(annotation_path).to_tlbr()
        samples.append(
            {
                "image_path": str(image_path),
                "boxes": boxes.array[:, :4],
                "labels": boxes.labels,
            }
        )
    return samples


class RetinaNetDataset(torch.utils.data.Dataset):  # type: ignore
    """Dataset class for RetinaNet model."""

    def __init__(self, holdout: Optional[Holdout] = None) -> None:
        self.samples = get_sample_dicts(holdout)
        self.transform = T.ToTensor()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get the tensor dict for a sample."""
        sample = self.samples[idx]
        img = Image.open(sample["image_path"])
        x = self.transform(img)
        boxes = sample["boxes"].astype("float32")
        return {
            "image": x,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(sample["labels"]),
        }
