import os

import cv2
from tqdm import tqdm

path = "/code/data/datasets/common_hall/demo_imgs_upload"

img_list = os.listdir(path)

for i in tqdm(img_list):
    img_path = os.path.join(path, i)
    img = cv2.imread(img_path)
    img = cv2.rectangle(img, (450, 200), (1280, 720), (0, 255, 0), thickness=4)
    cv2.imwrite(img_path, img)
