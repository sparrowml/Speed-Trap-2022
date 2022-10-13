import os
import random
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from speed_trapv3.config import Config
from speed_trapv3.detection.config import Config as DetConfig
from speed_trapv3.detection.model import RetinaNet
from speed_trapv3.keypoints.config import Config as KeyConfig
from speed_trapv3.keypoints.dataset import crop_and_resize
from speed_trapv3.keypoints.model import SegmentationModel
from speed_trapv3.utils import get_prediction_path, slugify


def keypoints_post_inference_processing(
    _keypoints, _resized_roi_w, _resized_roi_h, _roi_w, _roi_h, _x1, _y1
):
    cropped_resized_abs_keypoints = (
        _keypoints  # keypoints are in absolute space w.r.t the resized roi
    )
    cropped_resized_rel_keypoints = cropped_resized_abs_keypoints / np.array(
        [_resized_roi_w, _resized_roi_h]
    )  # convert to relative
    cropped_abs_keypoints = cropped_resized_rel_keypoints * np.array(
        [_roi_w, _roi_h]
    )  # keypoints are in absolute space w.r.t the roi dimensions (dimensions defined by the vehicle bounding box)
    abs_keypoints = cropped_abs_keypoints + np.array(
        [_x1, _y1]
    )  # keypoints are in absolute space w.r.t the original image size (Darwin file's dimensions)
    return abs_keypoints


detection_model = RetinaNet().eval().cuda()
detection_model.load(DetConfig.trained_model_path)

keypoint_model = SegmentationModel().eval().cuda()
keypoint_model.load(KeyConfig.trained_model_path)

image_transform = T.Compose([T.ToTensor()])

image_paths = list(Config.images_directory.glob("*.jpg"))
random.shuffle(image_paths)
score_threshold: float = 0.5
image_paths = image_paths[:3]
for image_path in tqdm(image_paths):
    slug = slugify(image_path)
    img = imageio.imread(image_path)
    img_h, img_w, _ = img.shape
    boxes = detection_model(img)
    boxes = boxes[boxes.scores > score_threshold].to_relative().to_tlbr()
    boxes = boxes.array[:, :4]
    for box in boxes:
        x1, y1, x2, y2 = (box * np.array([img_w, img_h, img_w, img_h])).astype(int)
        roi_w = x2 - x1
        roi_h = y2 - y1
        roi_resized = crop_and_resize(
            box, img, KeyConfig.image_crop_size[0], KeyConfig.image_crop_size[1]
        )
        roi_resized_w, roi_resized_h = roi_resized.size
        x = image_transform(roi_resized)
        x = torch.unsqueeze(x, 0).cuda()
        keypoints = keypoint_model(x)["keypoints"][0].detach().cpu().numpy()
        keypoints = keypoints_post_inference_processing(
            keypoints, roi_resized_w, roi_resized_h, roi_w, roi_h, x1, y1
        )
        img = cv2.rectangle(
            img, (x1, y1), (x2, y2), (255, 0, 255), thickness=4
        )  # Magenta: bounding boxes
        img = cv2.circle(
            img,
            (int(keypoints[0][0]), int(keypoints[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )  # Blue: Backtire
        img = cv2.circle(
            img,
            (int(keypoints[1][0]), int(keypoints[1][1])),
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
        )  # Red Front tire
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    save_path = "/code/data/datasets/temp_imgs"
    filename = str(slug) + ".jpg"
    cv2.imwrite(os.path.join(save_path, filename), im_rgb)

    # boxes.to_file(get_prediction_path(slug))
