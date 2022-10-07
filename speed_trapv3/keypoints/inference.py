import json
import random
import tempfile
from pathlib import Path

import numpy as np
from darwin import Client, importer
from PIL import Image
from tqdm import tqdm

from .config import Config
from .dataset import image_transform
from .model import SegmentationModel


def run_predictions() -> None:
    """Generate key point predictions for new images."""
    model = SegmentationModel().eval().cuda()
    model.load(Config.trained_model_path)
    slugs = [
        slug.name.removesuffix(".jpg")
        for slug in Path(Config.images_directory).glob("*.jpg")
    ]
    random.shuffle(slugs)
    resize_height, resize_width = Config.image_resize
    for slug in tqdm(slugs):
        image_path = Config.images_directory / f"{slug}.jpg"
        img = Image.open(image_path)
        raw_width, raw_height = img.size
        x = image_transform(img).cuda()
        result = model(x[None])
        keypoints = result["keypoints"][0].detach().cpu().numpy()
        keypoints = (
            np.array([raw_width, raw_height])
            * keypoints
            / np.array([resize_width, resize_height])
        )
        with open(Config.predictions_directory / f"{slug}.json", "w") as f:
            f.write(json.dumps(keypoints.astype(float).tolist()))


def import_predictions(empty_annotations: str) -> None:
    """Import keypoint predictions for images without annotations."""
    client = Client.local(Config.darwin_team_slug)
    dataset = client.get_remote_dataset(Config.darwin_dataset_slug)
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_paths = []
        for prediction_path in Config.predictions_directory.glob("*.json"):
            slug = prediction_path.name.removesuffix(".json")
            with open(prediction_path) as f:
                keypoints = json.loads(f.read())
            empty_annotation_path = Path(empty_annotations) / f"{slug}.json"
            try:
                with open(empty_annotation_path) as f:
                    darwin_data = json.loads(f.read())
                nodes = []
                for full_point_name, (x, y) in zip(Config.keypoint_names, keypoints):
                    _, sub_point_name = full_point_name.split("-")
                    nodes.append(
                        {
                            "name": sub_point_name,
                            "occluded": False,  # Hardcoded for now
                            "x": x,
                            "y": y,
                        }
                    )
                court_nodes = nodes[:4]
                net_nodes = nodes[4:]
                darwin_data["annotations"] = [
                    {"name": "court", "skeleton": {"nodes": court_nodes}},
                    {"name": "net", "skeleton": {"nodes": net_nodes}},
                ]
                predicted_annotation_path = Path(tmpdir) / f"{slug}.json"
                with open(predicted_annotation_path, "w") as f:
                    f.write(json.dumps(darwin_data))
                upload_paths.append(predicted_annotation_path)
                # print(len(upload_paths))
            except:
                continue
        importer.import_annotations(
            dataset,
            importer.get_importer("darwin"),
            upload_paths,
            append=False,
            class_prompt=False,
        )
