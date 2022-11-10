import io
import json
from pathlib import Path
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from sparrow_datums import AugmentedBoxTracking, BoxTracking
from tqdm import tqdm

from .config import Config as SpeedConfig


def validate_inclusion(_x, _y, _cx, _cy, _r):
    return (_x - _cx) ** 2 + (_y - _cy) ** 2 < _r**2


DETECTION_AREA_START_X = SpeedConfig.DETECTION_AREA_START_X
INCLUSION_RADIUS = SpeedConfig.INCLUSION_RADIUS


def write_results(speed_log_path, framewise_aggregation_path, gz_path, video_path_in):
    video_path = video_path_in
    f = open(speed_log_path, "r")
    speed_log = json.load(f)
    f.close()

    f = open(framewise_aggregation_path, "r")
    framewise_aggregation = json.load(f)
    f.close()

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    frame_border = True
    class_label: bool = False
    score_label: bool = False
    object_label: bool = True
    rule0 = True
    rule1 = True
    rule2 = True
    vehicle_chunk = AugmentedBoxTracking.from_box_tracking(
        BoxTracking.from_file(gz_path)
    )
    vehicle_tracklet_list = BoxTracking.from_file(gz_path).to_dict()["object_ids"]
    speed_log_vehice_ids = list(speed_log.keys())
    slug = Path(video_path_in).name.removesuffix(".mp4")
    with imageio.get_writer(
        SpeedConfig.json_directory / slug / f"{slug}_predictions.mp4",
        mode="I",
        fps=fps,
        macro_block_size=None,
    ) as writer:
        frame_idx = 0
        object_count_log = {}
        last_known_speed = {}
        vehiclewise_speed_record = {}
        for img, vehicle_boxes in tqdm(zip(reader, vehicle_chunk)):
            frame_log = {}
            frame_log["frame_idx"] = frame_idx
            frame_log["annotations"] = []
            boxes = vehicle_boxes  # vehicle_boxes is a len = 16 list where unavailable objects are nan.
            height, width = img.shape[:2]
            fig = plt.figure(
                frameon=False, figsize=(width / 100, height / 100), dpi=100
            )
            fig.add_axes((0, 0, 1, 1))
            plt.imshow(img)
            if frame_border:
                plt.plot(
                    [450, 1280, 1280, 450, 450],
                    [200, 200, 720, 720, 200],
                    lw=2,
                    c="green",
                )
            object_idx = 0
            for i, box in enumerate(boxes.to_absolute()):
                if not np.isfinite(box.x):
                    continue
                x1 = np.clip(box.x1, 2, width - 2)
                x2 = np.clip(box.x2, 2, width - 2)
                y1 = np.clip(box.y1, 2, height - 2)
                y2 = np.clip(box.y2, 2, height - 2)
                # rule #0
                if rule0 == True and (
                    x1 < 600 or x1 >= 1100
                ):  # if the object is located either in far-left or far-right, ignore it.
                    continue
                color: Optional[str] = None
                text_strings: list[str] = []
                if class_label:
                    text_strings.append(f"class: {int(box.label)}")
                    color = f"C{int(box.label)}"
                if score_label:
                    text_strings.append(f"score: {box.score:.2f}")
                if object_label:
                    if (
                        str(i) in speed_log_vehice_ids
                        and str(frame_idx) in speed_log[str(i)]
                        and speed_log[str(i)][str(frame_idx)] >= 0
                    ):
                        text_strings.append(f"object_id: {i}")
                        text_strings.append(
                            f"\n current speed: {speed_log[str(i)][str(frame_idx)]} mph"
                        )
                        if i in vehiclewise_speed_record:
                            vehiclewise_speed_record[i].append(
                                speed_log[str(i)][str(frame_idx)]
                            )
                        else:
                            vehiclewise_speed_record[i] = [
                                speed_log[str(i)][str(frame_idx)]
                            ]
                        last_known_speed[str(i)] = speed_log[str(i)][str(frame_idx)]
                    elif str(i) in speed_log_vehice_ids and str(i) in last_known_speed:
                        text_strings.append(f"object_id: {i}")
                        text_strings.append(
                            f"\n current speed: {last_known_speed[str(i)]} mph"
                        )
                    else:
                        text_strings.append(f"object_id: {i}")
                    if color is None:
                        color = f"C{i}"
                if color is None:
                    color = "C0"
                plt.text(
                    x1 + 3,
                    y1 - 8,
                    ", ".join(text_strings),
                    backgroundcolor=(1, 1, 1, 0.5),
                    c="black",
                    size=8,
                )
                plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=2, c=color)
                back_tire, front_tire = framewise_aggregation[frame_idx]["annotations"][
                    object_idx
                ]["keypoints"]
                back_tire_x, back_tire_y = back_tire
                front_tire_x, front_tire_y = front_tire
                keypoints_score = framewise_aggregation[frame_idx]["annotations"][
                    object_idx
                ]["keypoints_scores"]
                # rule #1 and rule #2
                # TF
                if rule1 == True and rule2 == False:
                    if back_tire_y > y1 + int(
                        (y2 - y1) / 2
                    ):  # if the tire located in the lower half of the bbx, plot it.
                        plt.plot(back_tire_x, back_tire_y, marker="o", color="red")
                    if front_tire_y > y1 + int((y2 - y1) / 2):
                        plt.plot(front_tire_x, front_tire_y, marker="o", color="blue")
                # FT
                if rule1 == False and rule2 == True:
                    if validate_inclusion(
                        back_tire_x,
                        back_tire_y,
                        front_tire_x,
                        front_tire_y,
                        INCLUSION_RADIUS,
                    ):
                        if np.array(keypoints_score).argmax() == 0:
                            plt.plot(back_tire_x, back_tire_y, marker="o", color="red")
                        else:
                            plt.plot(
                                front_tire_x, front_tire_y, marker="o", color="blue"
                            )
                    else:
                        plt.plot(back_tire_x, back_tire_y, marker="o", color="red")
                        plt.plot(front_tire_x, front_tire_y, marker="o", color="blue")
                # TT
                if rule1 == True and rule2 == True:
                    if back_tire_y > y1 + int((y2 - y1) / 2):
                        if validate_inclusion(
                            back_tire_x,
                            back_tire_y,
                            front_tire_x,
                            front_tire_y,
                            INCLUSION_RADIUS,
                        ):
                            if np.array(keypoints_score).argmax() == 0:
                                plt.plot(
                                    back_tire_x, back_tire_y, marker="o", color="red"
                                )
                            elif front_tire_y > y1 + int((y2 - y1) / 2):
                                plt.plot(
                                    front_tire_x, front_tire_y, marker="o", color="blue"
                                )
                        elif front_tire_y > y1 + int((y2 - y1) / 2):
                            plt.plot(back_tire_x, back_tire_y, marker="o", color="red")
                            plt.plot(
                                front_tire_x, front_tire_y, marker="o", color="blue"
                            )
                        else:
                            plt.plot(back_tire_x, back_tire_y, marker="o", color="red")
                    elif front_tire_y > y1 + int((y2 - y1) / 2):
                        plt.plot(front_tire_x, front_tire_y, marker="o", color="blue")

                # FF
                if rule1 == False and rule2 == False:
                    plt.plot(back_tire_x, back_tire_y, marker="o", color="red")
                    plt.plot(front_tire_x, front_tire_y, marker="o", color="blue")

                if (
                    x1 > DETECTION_AREA_START_X
                    and vehicle_tracklet_list[i] not in object_count_log
                ):
                    object_count_log[vehicle_tracklet_list[i]] = True
            object_idx += 1
            plt.text(
                100,
                100,
                f"Vehicle Count = {len(object_count_log)}",
                backgroundcolor=(1, 0.5, 1, 0.5),
                c="black",
                fontsize=16,
            )
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            frame = imageio.v2.imread(buffer.getbuffer(), format="png")
            writer.append_data(frame)
            frame_idx += 1
        vehiclewise_speed_record_summary = {}
        for object_id in vehiclewise_speed_record.keys():
            vehiclewise_speed_record_summary[object_id] = {
                "MaxSpeed": max(vehiclewise_speed_record[object_id]),
                "AvgSpeed": sum(vehiclewise_speed_record[object_id])
                / len(vehiclewise_speed_record[object_id]),
            }
        highest_speed = -np.inf
        fastest_vehicle = None
        total_vehicle_avg_speed = 0
        for object_id in vehiclewise_speed_record_summary:
            if vehiclewise_speed_record_summary[object_id]["MaxSpeed"] > highest_speed:
                highest_speed = vehiclewise_speed_record_summary[object_id]["MaxSpeed"]
                fastest_vehicle = object_id
            total_vehicle_avg_speed += vehiclewise_speed_record_summary[object_id][
                "AvgSpeed"
            ]
        average_vehicle_speed = int(
            total_vehicle_avg_speed / len(vehiclewise_speed_record_summary)
        )
        vehicle_count = len(object_count_log)
        final_summary = {
            "MaxSpeed": highest_speed,
            "FastestVehicle": fastest_vehicle,
            "AverageVehicleSpeed": average_vehicle_speed,
            "VehicleCount": vehicle_count,
        }
        f = open(SpeedConfig.json_directory / slug / "speed_report.json", "w")
        json.dump(final_summary, f)
        f.close()
