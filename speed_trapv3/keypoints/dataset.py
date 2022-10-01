import json
import os
from distutils.command.config import config
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .config import Config
from .utils import Holdout, get_holdout

image_transform = T.Compose(
    [
        T.Resize(Config.image_resize),
        T.ToTensor(),
        # T.Normalize(Config.rgb_mean, Config.rgb_std),
    ]
)


def version_annotations(darwin_path: str) -> None:
    """Convert Darwin annotations to Sparrow format so they can be versioned."""
    raw_annotations_directory = Path(darwin_path)
    slugs = set(
        [p.name.removesuffix(".json") for p in raw_annotations_directory.glob("*.json")]
    )
    total_annotations = 0
    no_labels = []
    for slug in slugs:
        points: dict[str, tuple[float, float]] = dict()
        annotation_path = raw_annotations_directory / f"{slug}.json"
        with open(annotation_path) as f:
            raw_data = json.loads(f.read())
        w = raw_data["image"]["width"]
        h = raw_data["image"]["height"]
        for annotation in raw_data["annotations"]:
            object_name = annotation["name"]
            if object_name in ("back_tire", "front_tire"):
                # Save relative points to disk
                x, y = map(
                    float,
                    [
                        annotation["keypoint"]["x"] / w,
                        annotation["keypoint"]["y"] / h,
                    ],
                )
                points[object_name] = x, y
        output = []
        for key in Config.keypoint_names:
            if key in points:
                output.append(points[key])
        if len(output) > 0:
            with open(Config.annotations_directory / f"{slug}.json", "w") as f:
                f.write(json.dumps(output))
            total_annotations += 1
        else:
            no_labels.append(slug)
    print(
        f"{total_annotations} annotation files saved at {Config.annotations_directory}"
    )
    if len(no_labels) > 0:
        print("Warning: these files did not have keypoints at all!", no_labels)


def keypoints_to_heatmap(
    x0: int, y0: int, w: int, h: int, covariance: float = Config.covariance_2d
) -> np.ndarray:
    """Create a 2D heatmap from an x, y pixel location."""
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    zz = (
        1
        / (2 * np.pi * covariance**2)
        * np.exp(
            -(
                (xx - x0) ** 2 / (2 * covariance**2)
                + (yy - y0) ** 2 / (2 * covariance**2)
            )
        )
    )
    # Normalize zz to be in [0, 1]
    zz_min = zz.min()
    zz_max = zz.max()
    zz_range = zz_max - zz_min
    if zz_range == 0:
        zz_range += 1e-8
    return (zz - zz_min) / zz_range


def get_sample_dicts(holdout: Optional[Holdout] = None) -> list[dict[str, Any]]:
    """Get sample dicts."""
    slugs = set(
        [
            p.name.removesuffix(".json")
            for p in Config.annotations_directory.glob("*.json")
        ]
    )
    samples = []

    for slug in slugs:
        sample_holdout = get_holdout(slug)
        if holdout and holdout != sample_holdout:
            continue
        image_path = Config.images_directory / f"{slug}.jpg"
        annotation_path = Config.annotations_directory / f"{slug}.json"
        with open(annotation_path) as f:
            keypoints = np.array(json.loads(f.read()))
            if (
                len(keypoints) < Config.num_classes
            ):  # if all classes aren't present, take the first class (row) and repeat it to compensate for the absent classes.
                repititions = Config.num_classes
                axis = 1
                keypoints = np.tile(keypoints[0], (repititions, axis))

        samples.append(
            {
                "holdout": sample_holdout.name,
                "image_path": str(image_path),
                "keypoints": keypoints,
                "labels": Config.keypoint_names,
            }
        )
    return samples


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset class for Segmentations model."""

    def __init__(self, holdout: Optional[Holdout] = None) -> None:
        self.samples = get_sample_dicts(holdout)

    def __len__(self):
        """Length of the sample."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get the tensor dict for a sample."""
        sample = self.samples[idx]
        resize_height, resize_width = Config.image_resize
        keypoints = sample["keypoints"] * np.array([resize_width, resize_height])
        heatmaps = []
        for x, y in keypoints:
            heatmaps.append(keypoints_to_heatmap(x, y, resize_width, resize_height))
        heatmaps = np.stack(heatmaps, 0)
        img = Image.open(sample["image_path"])
        x = image_transform(img)

        return {
            "holdout": sample["holdout"],
            "image_path": sample["image_path"],
            "heatmaps": heatmaps.astype("float32"),
            "keypoints": keypoints.astype("float32"),
            "labels": sample["labels"],
            "image": x,
        }


# version_annotations(
#     "/root/.darwin/datasets/sparrow-computing/kj_speedtrap/releases/backtirev1.0/annotations"
# )
