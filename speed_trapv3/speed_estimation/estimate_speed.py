import json
import os
from math import sqrt
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from tqdm import tqdm

from .config import Config as SpeedConfig


def validate_inclusion(_x, _y, _cx, _cy, _r):
    return (_x - _cx) ** 2 + (_y - _cy) ** 2 < _r**2


def get_distance(_x1, _y1, _x2, _y2):
    return sqrt((_x1 - _x2) ** 2 + (_y1 - _y2) ** 2)


def frames_to_seconds(_fps, _n_frames):
    return (1 / _fps) * _n_frames


def get_objectwise_keypoints(rule0, rule1, rule2, video_path_in):
    objectwise_keypoints = {}
    slug = Path(video_path_in).name.removesuffix(".mp4")
    f = open(SpeedConfig.json_directory / slug / "framewise_aggregation.json")
    frame_to_predictions_map = json.load(f)
    f = open(SpeedConfig.json_directory / slug / "objectwise_aggregation.json")
    objects_to_predictions_map = json.load(f)[
        "annotations"
    ]  # the object_id attribute of frame_to_predictions_map are the keys of object_to_predictions_map
    for i in range(len(objects_to_predictions_map)):
        objectwise_keypoints[objects_to_predictions_map[i]["id"]] = []
    for frame_idx in range(len(frame_to_predictions_map)):
        objects_per_frame = frame_to_predictions_map[frame_idx]["annotations"]
        for obj_idx in range(len(objects_per_frame)):
            back_tire_x = objects_per_frame[obj_idx]["keypoints"][0][0]
            back_tire_y = objects_per_frame[obj_idx]["keypoints"][0][1]
            front_tire_x = objects_per_frame[obj_idx]["keypoints"][1][0]
            front_tire_y = objects_per_frame[obj_idx]["keypoints"][1][1]
            x1, y1, x2, y2 = objects_per_frame[obj_idx]["bounding_box"]
            # rule #0
            if rule0 == True:
                if x1 < SpeedConfig.rule0_x_min or x1 >= SpeedConfig.rule0_x_max:
                    back_tire_x = -100
                    back_tire_y = -100
                    front_tire_x = -100
                    front_tire_y = -100

            # rule #1
            if rule1 == True:
                if back_tire_y <= y1 + int((y2 - y1) / 2):
                    back_tire_x = -100
                    back_tire_y = -100
                if front_tire_y <= y1 + int((y2 - y1) / 2):
                    front_tire_x = -100
                    front_tire_y = -100
            # rule #2
            if rule2 == True:
                if validate_inclusion(
                    back_tire_x,
                    back_tire_y,
                    front_tire_x,
                    front_tire_y,
                    SpeedConfig.rule1_min_tire_dist,
                ):
                    if (
                        np.array(
                            objects_per_frame[obj_idx]["keypoints_scores"]
                        ).argmax()
                        == 0
                    ):
                        front_tire_x = -100
                        front_tire_y = -100
                    else:
                        back_tire_x = -100
                        back_tire_y = -100
            objectwise_keypoints[
                objects_per_frame[obj_idx]["object_tracklet_id"]
            ].append((back_tire_x, back_tire_y, front_tire_x, front_tire_y))
    return objectwise_keypoints


def estimate_speed(video_path_in):
    video_path = video_path_in
    slug = Path(video_path).name.removesuffix(".mp4")
    f = open(SpeedConfig.json_directory / slug / "framewise_aggregation.json")
    frame_to_predictions_map = json.load(f)
    f = open(SpeedConfig.json_directory / slug / "objectwise_aggregation.json")
    objects_to_predictions_map = json.load(f)[
        "annotations"
    ]  # the object_id attribute of frame_to_predictions_map are the keys of object_to_predictions_map
    objectwise_keypoints = get_objectwise_keypoints(True, True, True, video_path)
    object_names = list(objectwise_keypoints.keys())
    #
    vehicle_keypoints = {}
    for obj_idx in range(len(object_names)):
        dont_care_count = 0
        object_name = object_names[obj_idx]
        noisy_vehicle_keypoints = objectwise_keypoints[object_names[obj_idx]]
        for vehicle_keypoint_pair in noisy_vehicle_keypoints:
            back_x, back_y, front_x, front_y = vehicle_keypoint_pair
            if back_x < 0 and front_x < 0:
                dont_care_count += 1
        dont_care_percentage = (
            dont_care_count * 100 / len(noisy_vehicle_keypoints)
        )  # calculate the noise percentage
        if dont_care_percentage == 100:  # throw away if noise percentage is 100
            continue
        else:
            vehicle_keypoints[obj_idx] = noisy_vehicle_keypoints
    vehicle_indices = list(vehicle_keypoints.keys())
    #
    speed_collection = {}
    for vehicle_index in vehicle_indices:  # Looping through all objects in the video
        approximate_speed = -1
        back_tire_x_list = []
        back_tire_y_list = []
        front_tire_x_list = []
        front_tire_y_list = []
        object_name = object_names[vehicle_index]
        for keypoints_per_frame in objectwise_keypoints[object_name]:
            back_tire_x, back_tire_y, front_tire_x, front_tire_y = keypoints_per_frame
            back_tire_x_list.append(back_tire_x)
            back_tire_y_list.append(back_tire_y)
            front_tire_x_list.append(front_tire_x)
            front_tire_y_list.append(front_tire_y)
        #
        data = {
            "back_tire_x": back_tire_x_list,
            "back_tire_y": back_tire_y_list,
            "front_tire_x": front_tire_x_list,
            "front_tire_y": front_tire_y_list,
        }
        df = pd.DataFrame(data)
        df.drop(df[df.back_tire_x < 0].index, inplace=True)
        df.drop(df[df.front_tire_x < 0].index, inplace=True)
        if len(df) == 0:  # If all the data are single points, do not filter it.
            data = {
                "back_tire_x": back_tire_x_list,
                "back_tire_y": back_tire_y_list,
                "front_tire_x": front_tire_x_list,
                "front_tire_y": front_tire_y_list,
            }
            df = pd.DataFrame(data)
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        #
        model = linear_model.LinearRegression()
        model.fit(x, y)
        coef = model.coef_[0]
        bias = model.intercept_
        coef = [
            i + 0.0000001 if i == 0 else i for i in coef
        ]  # Add a small value to avoid division by zero.
        #
        back_tire_x_list = []
        back_tire_y_list = []
        front_tire_x_list = []
        front_tire_y_list = []
        for i in range(len(data["back_tire_x"])):
            back_tire_x = data["back_tire_x"][i]
            back_tire_y = data["back_tire_y"][i]
            front_tire_x = data["front_tire_x"][i]
            front_tire_y = data["front_tire_y"][i]
            if (back_tire_x < 0 and back_tire_y < 0) and (
                front_tire_x < 0 and front_tire_y < 0
            ):
                back_tire_x_list.append(back_tire_x)
                back_tire_y_list.append(back_tire_y)
                front_tire_x_list.append(front_tire_x)
                front_tire_y_list.append(front_tire_y)
                continue
            if back_tire_x < 0:
                back_tire_x = (
                    front_tire_y - back_tire_y * coef[1] - front_tire_x * coef[2] - bias
                ) / coef[0]
            if back_tire_y < 0:
                back_tire_y = (
                    front_tire_y - back_tire_x * coef[0] - front_tire_x * coef[2] - bias
                ) / coef[1]
            if front_tire_x < 0:
                front_tire_x = (
                    front_tire_y - back_tire_x * coef[0] - back_tire_y * coef[1] - bias
                ) / coef[2]

            if front_tire_y < 0:
                front_tire_y = (
                    back_tire_x * coef[0]
                    + back_tire_y * coef[1]
                    + front_tire_x * coef[2]
                    + bias
                )

            back_tire_x_list.append(back_tire_x)
            back_tire_y_list.append(back_tire_y)
            front_tire_x_list.append(front_tire_x)
            front_tire_y_list.append(front_tire_y)
        #

        vehicle_speed = []
        skipped = 0

        back_tire_keypoints = [back_tire_x_list, back_tire_y_list]
        back_tire_keypoints = [list(x) for x in zip(*back_tire_keypoints[::-1])]
        front_tire_keypoints = [front_tire_x_list, front_tire_y_list]
        front_tire_keypoints = [list(x) for x in zip(*front_tire_keypoints[::-1])]

        back_tire_x_list = []
        back_tire_y_list = []
        front_tire_x_list = []
        front_tire_y_list = []
        speed_checkpoints = []
        #
        vehicle_speed = {}
        total_num_points = len(objectwise_keypoints[object_name])
        object_start = objects_to_predictions_map[vehicle_index]["segments"][0][0]
        for i in range(total_num_points):
            back_tire = back_tire_keypoints[i]
            front_tire = front_tire_keypoints[i]
            if back_tire[0] < 0 or front_tire[0] < 0:
                vehicle_speed[i + object_start] = approximate_speed
                skipped += 1
                continue
            for j in range(i, total_num_points):
                future_back_tire = back_tire_keypoints[j]
                if future_back_tire[0] < 0:
                    continue
                back_tire_x = back_tire[0]
                back_tire_y = back_tire[1]
                front_tire_x = front_tire[0]
                front_tire_y = front_tire[1]
                future_back_tire_x = future_back_tire[0]
                future_back_tire_y = future_back_tire[1]
                current_keypoints_distance = get_distance(
                    back_tire_x, back_tire_y, front_tire_x, front_tire_y
                )
                future_keypoints_distance = get_distance(
                    back_tire_x, back_tire_y, future_back_tire_x, future_back_tire_y
                )
                if current_keypoints_distance <= future_keypoints_distance:
                    if (
                        future_keypoints_distance - current_keypoints_distance
                    ) < SpeedConfig.distance_error_threshold:
                        approximate_speed = round(
                            SpeedConfig.MPERSTOMPH
                            * SpeedConfig.WHEEL_BASE
                            / frames_to_seconds(30, j - i)
                        )
                        vehicle_speed[i + object_start] = approximate_speed
                        back_tire_x_list.append(back_tire_x)
                        back_tire_y_list.append(back_tire_y)
                        front_tire_x_list.append(front_tire_x)
                        front_tire_y_list.append(front_tire_y)
                    break
        speed_collection[vehicle_index] = vehicle_speed
    f = open(SpeedConfig.json_directory / slug / "speed_log.json", "w")
    json.dump(speed_collection, f)
    f.close()
