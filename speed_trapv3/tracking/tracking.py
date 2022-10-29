"""Object tracking."""
import os
from pathlib import Path
from typing import Union

import cv2
import imageio
import numpy as np
import onnxruntime as ort
from PIL import Image
from sparrow_datums import AugmentedBoxTracking, BoxTracking, FrameBoxes, PType
from sparrow_tracky import Tracker, euclidean_distance
from tqdm import tqdm

from .config import Config


def transform_image(x: np.ndarray) -> np.ndarray:
    """Take raw image frame and pre-process for inference."""
    x = Image.fromarray(x)
    x = x.resize((Config.image_width, Config.image_height))
    x = np.array(x).astype("float32")
    return x.transpose(-1, 0, 1)[None]


def get_frame_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[FrameBoxes, FrameBoxes]:
    """Convert bounding boxes from numpy array into ball and player FrameBoxes.
    Parameters
    ----------
    boxes
        (n, 4) boxes in absolute tlbr format with shape Config.image_size
    scores
        Confidence that each box is an object
    labels
        Class indices (0 == ball, 1 == active player)
    Returns
    -------
    ball_boxes, player_boxes
        Output boxes with width/height corresponding to the input video
    """
    relative_boxes = boxes / np.array([Config.image_width, Config.image_height] * 2)
    vehicle_mask = np.logical_and(
        labels == Config.vehicle_class_index, scores > Config.vehicle_score_threshold
    )
    vehicle_boxes = FrameBoxes(
        relative_boxes[vehicle_mask],
        PType.relative_tlbr,
        image_width=image_width,
        image_height=image_height,
    )
    return vehicle_boxes


def get_video_properties(video_path: Union[str, Path]) -> tuple[float, int]:
    """
    Get the frames per second value and the number of frames of the input video.
    Parameters
    ----------
    video_path
        Location of the input video
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, n_frames


def make_path(path_in):
    """Check if a given path exists and make one if it doesn't.

    Args:
        path_in (str): _description_ Input path that might need to be created.
    """
    if not os.path.exists(path_in):
        os.mkdir(path_in)


def track_objects(
    video_path: Union[str, Path],
    model_path: Union[str, Path] = Config.onnx_model_path,
) -> None:
    """
    Track ball and the players in a video.
    Parameters
    ----------
    video_path
        The path to the source video
    output_path
        The path to write the chunk to
    model_path
        The path to the ONNX model
    """
    video_path = Path(video_path)
    slug = video_path.name.removesuffix(".mp4")
    vehicle_tracker = Tracker(Config.vehicle_iou_threshold)
    sess = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    fps, n_frames = get_video_properties(video_path)
    reader = imageio.get_reader(video_path)
    for i in tqdm(range(n_frames)):
        data = reader.get_data(i)
        data = cv2.rectangle(
            data, (450, 200), (1280, 720), thickness=5, color=(0, 255, 0)
        )
        input_height, input_width = data.shape[:2]
        x = transform_image(data)
        boxes, scores, labels = sess.run(None, {"input": x[0]})
        vehicle_boxes = get_frame_box(
            boxes, scores, labels, image_width=input_width, image_height=input_height
        )
        vehicle_tracker.track(vehicle_boxes)
    make_path(str(Config.prediction_directory / slug))
    vehicle_chunk = vehicle_tracker.make_chunk(fps, Config.vehicle_tracklet_length)
    vehicle_chunk.to_file(
        Config.prediction_directory / slug / f"{slug}_vehicle.json.gz"
    )


def write_to_json(chunk_path_in, video_path_in):
    """Convert the cunk into an AugmentedBoxTracking object and then write into a JSON file. This is a helper method to convert_to_darwin() method.

    Args:
        chunk_path_in (str): Location of the .gz files.
        video_path_in (str): Location of the input video.
    """
    chunk_path = chunk_path_in
    slug = chunk_path.name[:-8]
    filename = slug
    video_path = video_path_in
    chunk = BoxTracking.from_file(chunk_path).to_absolute().to_tlwh()
    aug_box = AugmentedBoxTracking.from_box_tracking(chunk)
    aug_box.to_darwin_file(
        output_path=Config.prediction_directory / f"objectwise_aggregation_{slug}.json",
        filename=filename,
        label_names=["vehicle"],
    )


def convert_to_darwin(path_in="/code/data/datasets/tracking/predictions"):
    """Write the chunk into a json.gz file.

    Args:
        path_in (str, optional): The source folder that contains the output of all the videos._description_. Defaults to "/code/data/datasets/tracking/predictions".
    """
    path = path_in
    n_skipped = 0
    n_created = 0
    n_total = 0
    video_name_list = os.listdir(path)
    for video_name in tqdm(video_name_list):
        video_path = os.path.join(path, video_name)
        for chunk_path in Path(video_path).rglob("*.json.gz"):
            n_total += 1
            try:
                write_to_json(chunk_path, video_path)
                n_created += 1
            except Exception as e:
                print(f"{str(chunk_path)} skipped due to chunk error")
                print(e)
                n_skipped += 1
                continue
    print("Total", n_total, "\n Created", n_created, "\n Total skipped", n_skipped)
