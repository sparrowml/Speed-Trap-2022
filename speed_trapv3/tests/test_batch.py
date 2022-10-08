import csv
from pathlib import Path

import cv2
import imageio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from sparrow_datums import BoxTracking
from torch.utils.data import DataLoader
from tqdm import tqdm

from speed_trapv3.keypoints.dataset import SegmentationDataset
from speed_trapv3.keypoints.utils import Holdout

val_data = SegmentationDataset(Holdout.DEV)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
temp_imgs = []
for batch in iter(val_dataloader):
    image = Image.open(batch["image_path"][0])
    plt.savefig(np.array(image))
    temp_imgs.append(batch)
