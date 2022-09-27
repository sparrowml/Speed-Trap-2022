import json
import os
from pathlib import Path

from .config import Config

# from speed_trapv3.keypoints import Config


def version_annotations(darwin_path: str) -> None:
    """Convert Darwin annotations to Sparrow format so they can be versioned."""
    raw_annotations_directory = Path(darwin_path)
    slugs = set(
        [p.name.removesuffix(".json") for p in raw_annotations_directory.glob("*.json")]
    )
    total_annotations = 0
    for slug in slugs:
        points: dict[str, tuple[float, float]] = dict()
        annotation_path = raw_annotations_directory / f"{slug}.json"
        with open(annotation_path) as f:
            raw_data = json.loads(f.read())
        w = raw_data["image"]["width"]
        h = raw_data["image"]["height"]
        for annotation in raw_data["annotations"]:
            object_name = annotation["name"]

            # if object_name not in ("back_tire"):
            #     continue
            # for node in annotation["skeleton"]["nodes"]:
            #     node_name = node["name"]
            #     name = f"{object_name}-{node_name}"
            # Save relative points to disk
            x, y = map(
                float,
                [annotation["keypoint"]["x"] / w, annotation["keypoint"]["y"] / h],
            )
            points[object_name] = x, y
        output = []
        for key in Config.keypoint_names:
            output.append(points[key])
        with open(Config.annotations_directory / f"{slug}.json", "w") as f:
            f.write(json.dumps(output))
        total_annotations += 1
    print(
        f"{total_annotations} annotation files saved at {Config.annotations_directory}"
    )


# version_annotations(
#     "/root/.darwin/datasets/sparrow-computing/kj_speedtrap/releases/backtirev1.0/annotations"
# )
