import base64
import gc
import json
import os
from pathlib import Path
from random import random
from sys import _current_frames
from typing import Any, Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm

from .config import Config


def get_cap(
    _path: str = "/code/data/datasets/source_video",
    _filename: str = "WBuBqS9h8.mp4",
):
    return cv2.VideoCapture(os.path.join(_path, _filename))


def get_frame_idx(
    _video_name: str = "WBuBqS9h8_",
    _n_frames: int = 1800,
    _path: str = "data/datasets/images",
):
    n_frames = _n_frames
    train_set = os.listdir(_path)
    filename = ""
    while filename == "" or filename in train_set:
        frame_idx = round(n_frames * np.random.uniform())
        filename = f"{_video_name}{filename}{frame_idx}.jpg"
    return (filename, frame_idx)


def sample_video_frames(
    _cap: object = None,
) -> None:
    """Sample a random frame from each video."""
    cap = _cap
    image_name, frame_idx = get_frame_idx(_n_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _, frame = cap.read()
    frame = cv2.rectangle(
        frame, (450, 200), (1280, 720), thickness=5, color=(0, 255, 0)
    )
    cv2.imwrite(str(Config.images_directory / image_name), frame)
    with open("inference.txt", "a") as f:
        f.write(str(Config.images_directory / image_name) + "\n")
