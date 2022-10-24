import io
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sparrow_datums import AugmentedBoxTracking, BoxTracking, FrameAugmentedBoxes
from tqdm import tqdm


def plot_boxes(
    image: npt.NDArray[np.uint8],
    boxes_in: FrameAugmentedBoxes,
    class_label: bool = False,
    score_label: bool = False,
    object_label: bool = False,
) -> io.BytesIO:
    """Plot boxes on an image."""
    height, width = image.shape[:2]
    fig = plt.figure(frameon=False, figsize=(width / 100, height / 100), dpi=100)
    fig.add_axes((0, 0, 1, 1))
    plt.imshow(image)
    for boxes in tqdm(boxes_in):
        for i, box in enumerate(boxes.to_absolute()):
            if not np.isfinite(box.x):
                continue
            x1 = np.clip(box.x1, 2, width - 2)
            x2 = np.clip(box.x2, 2, width - 2)
            y1 = np.clip(box.y1, 2, height - 2)
            y2 = np.clip(box.y2, 2, height - 2)
            color: Optional[str] = None
            text_strings: list[str] = []
            if class_label:
                text_strings.append(f"class: {int(box.label)}")
                color = f"C{int(box.label)}"
            if score_label:
                text_strings.append(f"score: {box.score:.2f}")
            if object_label:
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
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    return buffer


def tracking(video_path_in, vehicle_boxes_path_in, output_path_in):
    """Visualize a object tracking boxes for a video."""
    reader = imageio.get_reader(video_path_in)
    fps = reader.get_meta_data()["fps"]
    vehicle_chunk = AugmentedBoxTracking.from_box_tracking(
        BoxTracking.from_file(vehicle_boxes_path_in)
    )
    if vehicle_chunk.fps != fps:
        vehicle_chunk = vehicle_chunk.resample(fps)
    with imageio.get_writer(
        output_path_in, mode="I", fps=fps, macro_block_size=None
    ) as writer:
        for img, vehicle_boxes in tqdm(zip(reader, vehicle_chunk)):
            boxes = list(vehicle_boxes)
            result = plot_boxes(img, boxes, object_label=True)
            frame = imageio.v2.imread(result.getbuffer(), format="png")
            writer.append_data(frame)


tracking(
    "/code/data/datasets/common_hall/source_video/25_resampled_vid.mp4",
    "/code/data/datasets/tracking/predictions/25_resampled_vid_vehicle.json.gz",
    "/code/data/datasets/common_hall/tracking_output_videos",
)
