import glob
import json
from ctypes import Union
from distutils.command.config import config
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sparrow_datums import Boxes, PType

from .config import Config
from .utils import Holdout, get_holdout

image_transform = T.Compose(
    [
        # T.Resize(Config.image_resize),
        T.ToTensor()
    ]
)


class PrepAnnotations:
    """Set of tools to process the annotation files."""

    def __init__(self):
        self.vehicles = []
        self.back_tires = []
        self.front_tires = []
        self.vehicle_to_tires = {}
        self.completed_vehicles = {}
        self.img_dim = None

    def set_img_dim(self, _dim: tuple) -> None:
        """Set the image size to the size found in the JSON file.

        Parameters
        ----------
        _dim :
            Dimensions of the image
        """
        self.img_dim = _dim

    def get_img_dim(self) -> tuple:
        """Fetch the image dimensions

        Returns
        -------
            Dimensions of the image
        """
        return self.img_dim

    def validate_inclusion(self, _keypoint: Dict, _bbx) -> bool:
        """Check if a given keypoint is inside a bounding box.

        Parameters
        ----------
        _keypoint :
            Keypoint to be matched the keypoint
        _bbx :
            Bounding box to matched with the keypoint

        Returns
        -------
            Returns True if the keypoint is inside or on the boundary of the bounding box
        """
        return cv2.pointPolygonTest(_bbx, _keypoint, False) >= 0

    def preprocess_bbx(self, _bbx, form: Optional[str] = "tlbr"):
        """Preprocess the box to be compatible with openCV and file save.
        When form = tlbr, the bounding box is returned as a list of [x1, y1, x2, y2].
        Otherwise, it returns as a np.array([(x1, y1), (x1, y2),(x2, y2),(x2, y1)]).

        Returns
        -------
            Box coordinates are changed from absolute tlwh to relative tlbr.
        """
        w, h = self.get_img_dim()
        bbx = _bbx["bounding_box"]
        bbx = Boxes(
            np.array([bbx["x"], bbx["y"], bbx["w"], bbx["h"]]).round(0).astype(int),
            PType.absolute_tlwh,
        ).to_tlbr()
        bbx = bbx.array
        if form == "tlbr":  # used when writing to JSON files
            bbx = bbx / np.array([w, h, w, h])  # change to relative space
            return bbx.tolist()
        else:  # Used when interacting with openCV
            return np.array(
                [(bbx[0], bbx[1]), (bbx[0], bbx[3]), (bbx[2], bbx[3]), (bbx[2], bbx[1])]
            )

    def preprocess_keypoint(self, _keypoint: Dict, _relative: Optional[bool] = False):
        """Preprocess a given keypoint to write to file

        Parameters
        ----------
        _keypoint :
            Keypoint to be preprocessed
        _relative :
            Enable this flag if the coordinates need to be in relative space

        Returns
        -------
            A tuple of (x, y) coordinates of the processed keypoints
        """
        keypoint = _keypoint["keypoint"]
        if _relative == False:  # Used for openCV
            return (int(keypoint["x"]), int(keypoint["y"]))
        else:  # Used for writing to file
            w, h = self.get_img_dim()
            return (keypoint["x"] / w, keypoint["y"] / h)

    def find_relationships(self, _vehicles: list, _tires: list) -> None:
        """Map the keypoints found in Darwin annotation files to corresponding vehicles.
        Assumption: Every vehicle has at most two keypoints (front_tire and back_tire)

        Parameters
        ----------
        _vehicles :
            list of vehicles detected on the image
        _tires : list
            list of tires detected on the image
        """
        for tire in _tires:
            for vehicle in _vehicles:
                if vehicle["id"] in self.completed_vehicles:
                    continue
                if self.validate_inclusion(
                    self.preprocess_keypoint(tire), self.preprocess_bbx(vehicle, None)
                ):
                    if vehicle["id"] not in self.vehicle_to_tires:
                        self.vehicle_to_tires[vehicle["id"]] = {
                            "bounding_box": self.preprocess_bbx(vehicle, "tlbr"),
                            tire["name"]: self.preprocess_keypoint(tire, True),
                        }
                    else:
                        self.vehicle_to_tires[vehicle["id"]].update(
                            {tire["name"]: self.preprocess_keypoint(tire, True)}
                        )
                        self.completed_vehicles[vehicle["id"]] = True

    def get_vehicle_to_tires(self) -> Dict:
        """Fetch the global dictionary that maps vehicles to tires

        Returns
        -------
            A key value pair that keeps track of which tire belongs to which vehicle
        """
        return self.vehicle_to_tires


def version_annotations(_annotation_dir: str, _save_dir: str) -> None:
    """Convert Darwin annotations to Sparrow format so they can be versioned."""
    annotation_dir = _annotation_dir
    save_dir = _save_dir
    raw_annotations_directory = Path(annotation_dir)
    slugs = set(
        [p.name.removesuffix(".json") for p in raw_annotations_directory.glob("*.json")]
    )
    for slug in slugs:
        prep_tools = PrepAnnotations()
        annotation_path = raw_annotations_directory / f"{slug}.json"
        with open(annotation_path) as f:
            raw_data = json.loads(f.read())
        prep_tools.set_img_dim(
            (raw_data["image"]["width"], raw_data["image"]["height"])
        )  # Grab the image size from the JSON file and store it in a member of the PrepAnnotation class
        for annotation in raw_data["annotations"]:
            object_name = annotation["name"]
            if object_name == "vehicle":
                prep_tools.vehicles.append(annotation)
            elif object_name == "front_tire":
                prep_tools.front_tires.append(annotation)
            elif object_name == "back_tire":
                prep_tools.back_tires.append(annotation)
        prep_tools.find_relationships(
            prep_tools.vehicles, prep_tools.back_tires
        )  # map the back_tires to vehicles
        prep_tools.find_relationships(
            prep_tools.vehicles, prep_tools.front_tires
        )  # map the front_tires to vehicles
        vehicle_to_tires = prep_tools.get_vehicle_to_tires()
        for vehicle_id in vehicle_to_tires:
            vehicle = vehicle_to_tires[vehicle_id]
            name = f"{slug}--{vehicle_id}"
            with open(Path(save_dir) / f"{name}.json", "w") as f:
                f.write(json.dumps(vehicle))


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
    per_object_slugs = set(
        [
            p.name.removesuffix(".json")
            for p in Config.annotations_directory.glob("*.json")
        ]
    )
    samples = []

    for per_object_slug in per_object_slugs:
        slug = per_object_slug.split("--")[
            0
        ]  # Remove the vehicle_id assigned by Darwin
        sample_holdout = get_holdout(per_object_slug)
        if holdout and holdout != sample_holdout:
            continue
        image_path = Config.images_directory / f"{slug}.jpg"
        annotation_path = Config.annotations_directory / f"{per_object_slug}.json"
        keypoints = []
        with open(annotation_path) as f:
            annotation_content = json.loads(f.read())
            for tire in Config.keypoint_names:
                if tire in annotation_content:
                    keypoints.append(annotation_content[tire])
                else:
                    keypoints.append(np.array([0, 0]))  # pad with fake data
        samples.append(
            {
                "holdout": sample_holdout.name,
                "image_path": str(image_path),
                "keypoints": np.array(keypoints),
                "labels": Config.keypoint_names,
                "bounding_box": annotation_content["bounding_box"],
                "annotation_path": str(annotation_path),
            }
        )
    return samples


def crop_and_resize(_bbx, _img, _crop_width, _crop_height):
    bbx = _bbx
    img = np.array(_img)
    crop_width = _crop_width
    crop_height = _crop_height
    img_height, img_width, _ = img.shape
    img_size = np.array([img_width, img_height, img_width, img_height])
    x1, y1, x2, y2 = (bbx * img_size).astype(int)
    img = img[y1:y2, x1:x2]
    img = cv2.resize(img, (crop_width, crop_height))
    return Image.fromarray(img.astype(np.uint8))


def process_keypoints(_keypoints, _bbx):
    bbx = _bbx * np.array(
        [
            Config.original_image_size[0],
            Config.original_image_size[1],
            Config.original_image_size[0],
            Config.original_image_size[1],
        ]
    ).astype(int)
    x1, y1, x2, y2 = bbx
    roi_h = y2 - y1
    roi_w = x2 - x1
    keypoints = _keypoints * np.array(
        [Config.original_image_size[0], Config.original_image_size[1]]
    )
    shifted_keypoints = keypoints - np.array([[x1, y1], [x1, y1]])
    shifted_relative_keypoints = shifted_keypoints / np.array([roi_w, roi_h])

    return (
        shifted_relative_keypoints
        * np.array([Config.image_crop_size[0], Config.image_crop_size[1]])
    ).astype(int)


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
        # resize_height, resize_width = Config.image_resize
        # keypoints = sample["keypoints"] * np.array([resize_width, resize_height])
        # bbx = (
        #     sample["bounding_box"]
        #     * np.array([resize_width, resize_height, resize_width, resize_height])
        # ).astype(int)
        crop_width, crop_height = Config.image_crop_size
        keypoints = process_keypoints(sample["keypoints"], sample["bounding_box"])
        heatmaps = []
        for x, y in keypoints:
            heatmaps.append(keypoints_to_heatmap(x, y, crop_width, crop_height))
        heatmaps = np.stack(heatmaps, 0)
        img = Image.open(sample["image_path"])
        # img = maskup(img, bbx)
        img = crop_and_resize(sample["bounding_box"], img, crop_width, crop_height)
        x = image_transform(img)

        return {
            "holdout": sample["holdout"],
            "image_path": sample["image_path"],
            "annotation_path": sample["annotation_path"],
            "heatmaps": heatmaps.astype("float32"),
            "keypoints": keypoints.astype("float32"),
            "labels": sample["labels"],
            "image": x,
        }
