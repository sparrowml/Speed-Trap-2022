import io
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from sparrow_datums import AugmentedBoxTracking, Boxes, BoxTracking, FrameBoxes, PType
from tqdm import tqdm

from ..keypoints.config import Config as KeyConfig
from ..keypoints.dataset import (
    crop_and_resize,
    keypoints_post_inference_processing,
    process_keypoints,
)
from ..keypoints.model import SegmentationModel
from .config import Config as TrackConfig
from .tracking import get_video_properties, transform_image, write_to_json


def add_keypoints(video_path_in, chunk_path_in):
    slug = Path(video_path_in).name.removesuffix(".mp4")
    keypoint_model = SegmentationModel().eval().cuda()
    keypoint_model.load(KeyConfig.trained_model_path)
    reader = imageio.get_reader(video_path_in)
    # fps = reader.get_meta_data()["fps"]
    vehicle_chunk = AugmentedBoxTracking.from_box_tracking(
        BoxTracking.from_file(chunk_path_in)
    )
    vehicle_tracklet_list = BoxTracking.from_file(chunk_path_in).to_dict()["object_ids"]
    image_transform = T.Compose([T.ToTensor()])
    aggregated_predictions = []  # Len is equal to to the number of frames.
    frame_idx = 0
    for img, vehicle_boxes in tqdm(zip(reader, vehicle_chunk)):
        frame_log = {}
        frame_log["frame_idx"] = frame_idx
        frame_log["annotations"] = []
        boxes = vehicle_boxes  # vehicle_boxes is a len = 16 list where unavailable objects are nan.
        height, width = img.shape[:2]
        for i, box in enumerate(boxes.to_absolute()):
            object_log = {}
            if not np.isfinite(box.x):
                continue
            x1 = np.clip(box.x1, 2, width - 2)
            x2 = np.clip(box.x2, 2, width - 2)
            y1 = np.clip(box.y1, 2, height - 2)
            y2 = np.clip(box.y2, 2, height - 2)

            roi_w = x2 - x1
            roi_h = y2 - y1
            roi_resized = crop_and_resize(
                box.to_relative().array[:4],
                img,
                KeyConfig.image_crop_size[0],
                KeyConfig.image_crop_size[1],
            )
            roi_resized_w, roi_resized_h = roi_resized.size
            x = image_transform(roi_resized)
            x = torch.unsqueeze(x, 0).cuda()
            keypoints = keypoint_model(x)["keypoints"][0].detach().cpu().numpy()
            heatmaps = keypoint_model(x)["heatmaps"].detach().cpu()
            keypoints_scores = list(
                np.amax(torch.flatten(heatmaps, 2).numpy(), axis=-1)[0]
            )
            keypoints = keypoints_post_inference_processing(
                keypoints, roi_resized_w, roi_resized_h, roi_w, roi_h, x1, y1
            )
            object_log["keypoints"] = [list(keypoints[0]), list(keypoints[1])]
            object_log["keypoints_scores"] = [
                keypoints_scores[0].item(),
                keypoints_scores[1].item(),
            ]
            object_log["object_id"] = i
            object_log["object_tracklet_id"] = vehicle_tracklet_list[i]
            object_log["bounding_box"] = list(box.array[:4])
            frame_log["annotations"].append(object_log)
            plt.plot(keypoints[0][0], keypoints[0][1], marker="o", color="red")
            plt.plot(keypoints[1][0], keypoints[1][1], marker="o", color="blue")
        aggregated_predictions.append(frame_log)
        frame_idx += 1
    out_file = open(
        str(TrackConfig.prediction_directory / slug / "framewise_aggregation.json"), "w"
    )
    json.dump(aggregated_predictions, out_file)
    out_file.close()
