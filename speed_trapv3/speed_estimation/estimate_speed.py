import json
import math
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import linear_model
from tqdm import tqdm

from .config import Config as SpeedConfig


def get_angle(_x1, _y1, _x2, _y2):
    """Calculates the angle between the stright line formed by x1, y1, x2, y2 and the horizontal line cuts through x1, y1 to find it's complementary angle.

    Parameters
    ----------
    _x1 : float
        x coordinate of the origin of the stright line
    _y1 : float
        y coordinate of the origin of the stright line
    _x2 : float
        x coordinate of the end of the stright line
    _y2 : float
        y coordinate of the end of the stright line

    Returns
    -------
    float
        complementary angle in degrees
    """
    if (_x2 - _x1) != 0:
        return 90 - math.degrees(math.atan((_y2 - _y1) / (_x2 - _x1)))
    else:
        return 90 - math.degrees(
            math.atan((_y2 - _y1) / ((_x2 - _x1) + 0.000001))
        )  # avoid zero diviion error


def validate_inclusion(_x, _y, _cx, _cy, _r):
    """Decides if a given point is inside the given radius.

    Parameters
    ----------
    _x : float
        x coordinate of the given point
    _y : float
        y coordinate of the given point
    _cx : float
        x coordinate of the center
    _cy : float
        y coordinate of the center
    _r : float
        radius of the circle.

    Returns
    -------
    bool
        Indicates if a point is within the circle or not.
    """
    return (_x - _cx) ** 2 + (_y - _cy) ** 2 < _r**2


def get_distance(_x1, _y1, _x2, _y2):
    """Finds the distance between two points.

    Parameters
    ----------
    _x1 : float
        x coordinate point 1
    _y1 : float
        y coordinate of point 1
    _x2 : float
        x coordinate of point 2
    _y2 : float
        y coordinate of point 2

    Returns
    -------
    float
        distance between two points
    """
    return sqrt((_x1 - _x2) ** 2 + (_y1 - _y2) ** 2)


def frames_to_seconds(_fps, _n_frames):
    """Converts frames to seconds given the frames per second property of a video.

    Parameters
    ----------
    _fps : int
        frame rate of the video
    _n_frames : int
        number of frames

    Returns
    -------
    float
        time elapsed in seconds
    """
    return (1 / _fps) * _n_frames


def get_objectwise_keypoints(rule0, rule1, rule2, video_path_in):
    """Here, we apply 3 rules to filter out noise in the keypoints predictions.

    Parameters
    ----------
    rule0 : bool
        Remove any keypoints in vehicles where x1 <600 or x1 >= 1100
    rule1 : bool
        If a keypoint is located in the upper half of the bounding box, remove it.
    rule2 : bool
        Draw a circle around the front tire keypoint and if the back tire is located inside of the circle, keep the tire that has the highest sigmoid score. Throw the other one away!!
    video_path_in : str
        _description_

    Returns
    -------
    list
        A list of dictionaries that include which vehicles appeared in each frame, including it's tire locations.
    """
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


def open_objects_to_predictions_map(slug):
    """Opens objectwise_aggregation.json

    Parameters
    ----------
    slug : str
        slug of the video

    Returns
    -------
    list
        A list of framewise information of the video.
    """
    f = open(SpeedConfig.json_directory / slug / "objectwise_aggregation.json")
    return json.load(f)[
        "annotations"
    ]  # the object_id attribute of frame_to_predictions_map are the keys of object_to_predictions_map


def filter_bad_tire_pairs(video_path_in):
    """Keypoints that violated the rules were marked as (-100, -100) in get_objectwise_keypoints() method. Here, we filter out keypoints where both back and front tire are (-1, -1).

    Parameters
    ----------
    video_path_in : str
        Path of the input video
        Returns
    -------
    Union[list, dict]
        All the object_tracklet_ids appeared in the video, object_id of the vehicles that remained after applying the rules, All the objectwise keypoints after rules were applied.
    """
    video_path = video_path_in
    objectwise_keypoints = get_objectwise_keypoints(True, True, True, video_path)
    object_names = list(objectwise_keypoints.keys())
    #
    vehicle_keypoints = {}
    for obj_idx in range(len(object_names)):
        dont_care_count = 0
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
    return object_names, list(vehicle_keypoints.keys()), objectwise_keypoints


def straight_line_fit(objectwise_keypoints, object_name):
    """Fit the good keypoints of a given object (vehicle) into a stright line.

    Parameters
    ----------
    objectwise_keypoints : Dict
        All the keypoints after the rules have been applied.
    object_name : list
        Name of the object that we are currently processing.

    Returns
    -------
    Union[list, dict]
        coefficients of the trained linear regression model, bias of that model, data that was used to train the model.
    """
    back_tire_x_list = []
    back_tire_y_list = []
    front_tire_x_list = []
    front_tire_y_list = []
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
    return coef, bias, data


def fill_missing_keypoints(coef, bias, data):
    """Fill in missing values of single tires.

    Parameters
    ----------
    coef : list
        Coefficients of the linear regression model
    bias : float
        bias of the model
    data : Dict
        Data that was used to train the model.

    Returns
    -------
    Tuple of lists
        returns four lists representing all four coordinates of the tires.
    """
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
    return back_tire_x_list, back_tire_y_list, front_tire_x_list, front_tire_y_list


def estimate_speed(video_path_in):
    """Estimate the speed of the vehicles in the video.

    Parameters
    ----------
    video_path_in : str
        Source video path
    """
    video_path = video_path_in
    slug = Path(video_path).name.removesuffix(".mp4")
    objects_to_predictions_map = open_objects_to_predictions_map(slug)
    object_names, vehicle_indices, objectwise_keypoints = filter_bad_tire_pairs(
        video_path
    )
    speed_collection = {}
    for vehicle_index in vehicle_indices:  # Looping through all objects in the video
        approximate_speed = -1
        object_name = object_names[vehicle_index]
        coef, bias, data = straight_line_fit(objectwise_keypoints, object_name)
        (
            back_tire_x_list,
            back_tire_y_list,
            front_tire_x_list,
            front_tire_y_list,
        ) = fill_missing_keypoints(coef, bias, data)

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
        # Speed calculation algorithm starts here...
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
                # if current_keypoints_distance <= future_keypoints_distance:
                if (
                    current_keypoints_distance - future_keypoints_distance
                ) >= -SpeedConfig.distance_error_threshold and (
                    current_keypoints_distance - future_keypoints_distance
                ) <= SpeedConfig.distance_error_threshold:
                    alpha = get_angle(
                        back_tire_x, back_tire_y, future_back_tire_x, future_back_tire_y
                    )
                    beta = get_angle(
                        back_tire_x, back_tire_y, front_tire_x, front_tire_y
                    )
                    if (
                        # (future_keypoints_distance - current_keypoints_distance)
                        # < SpeedConfig.distance_error_threshold
                        SpeedConfig.in_between_angle >= alpha + beta
                        and (j - i) > 1
                    ):
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
