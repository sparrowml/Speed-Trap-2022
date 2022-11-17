import gc
import os
from math import floor
from typing import Any, Optional

import cv2
import imageio
import numpy as np
from tqdm import tqdm
from typing_extensions import Self

from speed_trapv3.tracking.tracking import get_video_properties

from .config import Config


def video_to_frames(_video_path, _save_path):
    """Save all the frame of a given video as images.

    Parameters
    ----------
    _video_path : str
        src video
    _save_path : str
        dst of the video
    """
    reader = imageio.get_reader(_video_path)
    _, n_frames = get_video_properties(_video_path)
    for i in tqdm(range(n_frames)):
        frame = reader.get_data(i)
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        filename = f"{i}.jpg"
        cv2.imwrite(os.path.join(_save_path, filename), im_rgb)


class RandomResampleVideoToImages:
    def __init__(
        self,
        _src="/code/data/datasets/common_hall/source_video",
        _filename="WBuBqS9h8.mp4",
    ) -> None:
        self.src = _src
        self.filename = _filename

    def get_source_path(self):
        return self.src

    def get_filename(self):
        return self.filename

    def get_cap(self):
        return cv2.VideoCapture(
            os.path.join(self.get_source_path(), self.get_filename())
        )

    def get_slug(self):
        return self.get_filename.split(".")[0]

    def find_total_frames(self, video_in):
        """Calculate the total number of frames in the video."""
        cap = video_in
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_idx(
        self,
        _path: str = str(Config.images_directory),
    ):
        """Renders a randomly picked frame index that doesn't exist in the given image directory.

        Parameters
        ----------
        _path : str, optional
            Path of the exisisting images, by default str(Config.images_directory)

        Returns
        -------
        (filename, frame_idx) : Tuple
            Returns a tuple of filename and frame index.
        """
        n_frames = self.find_total_frames(self.get_cap())
        train_set = os.listdir(_path)
        filename = ""
        while filename == "" or filename in train_set:
            filename = ""
            frame_idx = round(n_frames * np.random.uniform())
            filename = f"{self.get_slug()}-{filename}-{frame_idx}.jpg"
        return (filename, frame_idx)

    def sample_video_frames(
        self,
        _cap: object = None,
    ) -> None:
        """Sample a random frame from each video."""
        cap = _cap
        image_name, frame_idx = self.get_frame_idx(
            _n_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()
        frame = cv2.rectangle(
            frame, (450, 200), (1280, 720), thickness=5, color=(0, 255, 0)
        )
        cv2.imwrite(str(Config.images_directory / image_name), frame)


class ResampleVideoToVideo:
    def __init__(
        self,
        _src="/code/data/datasets/common_hall/source_video/WBuBqS9h8.mp4",
        _dst="/code/data/datasets/common_hall/source_video",
        _percentile=50,
        _duration_in_sec=20,
    ) -> None:
        self.MAX_PERCENTILE = 100
        self.src = _src
        self.dst = _dst
        self.percentile = _percentile
        self.VIDEO_DURATION_SEC = _duration_in_sec

    def get_percentile(self):
        return self.percentile

    def get_destination_path(self):
        return self.dst

    def get_source_path(self):
        return self.src

    def get_max_percentile(self):
        return self.MAX_PERCENTILE

    def get_clip_duration_sec(self):
        return self.VIDEO_DURATION_SEC

    def find_total_frames(self, video_in, duration_in):
        """Calculate the total number of frames in the video."""
        return int(duration_in * int(video_in.get(cv2.CAP_PROP_FPS)))

    def find_start_frame(self, percentile_in, frames_in, duration_in):
        """
        Calculate the starting frame.
        :param percentile_in: desired instance of the video as a percentile.
        :param: frames_in: total number of frames
        :param: duraton_in: total duration of the video in seconds
        :return: starting frame coresponding to the given frame.
        """
        percentile = percentile_in
        total_frames = frames_in
        whole_video_duration_sec = duration_in
        start_frame = floor(percentile * total_frames / 100) - floor(
            self.get_clip_duration_sec() * total_frames / whole_video_duration_sec
        )
        if start_frame >= 0:
            return start_frame
        else:
            return 0

    def find_end_frame(self, percentile_in, frames_in):
        """
        Calculate the ending frame.
        :param percentile_in: desired instance of the video as a percentile.
        :param: frames_in: total number of frames
        :return: ending frame coresponding to the given frame.
        """
        percentile = percentile_in
        total_frames = frames_in
        return floor(percentile * total_frames / 100)

    def find_frame_list(self, cap_in, start_in, end_in):
        """
        Accumulate all the frames in the video to a python list.
        :param cap_in: videocapture object of the 1 min clip.
        :param start_in: starting frame of the clip.
        :param end_in: ending frame of the clip.
        :return: All video frames within a minute as a list.
        """
        cap = cap_in
        start_frame = start_in
        end_frame = end_in

        frame_list = []
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            else:
                frame_list.append(frame)
            frame_idx += 1
        return frame_list

    def find_whole_video_duration(self, _cap):
        """_summary_

         Parameters
         ----------
         _cap :
             cap object of the input video

         Returns
         -------
        float
             Duration of the input video in seconds
        """
        return round(_cap.get(cv2.CAP_PROP_FRAME_COUNT) / _cap.get(cv2.CAP_PROP_FPS))

    def write_to_mp4(self, cap_in, name_in, frames_in):
        """
        Write the video into an MP4 file and save it.
        :param: cap_in: videocapture object
        :param: name_in: slug
        :param: frames_in: A list of frames to be written into a video.
        :return: None
        """
        cap = cap_in
        save_name = str(os.path.join(self.get_destination_path(), name_in + ".mp4"))
        frame_list = frames_in
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(
            save_name, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height)
        )
        for i in range(len(frame_list)):
            frame = cv2.rectangle(
                frame_list[i], (450, 200), (1280, 720), (0, 255, 0), thickness=4
            )
            out.write(frame)
        out.release()

    def produce_resampled_video(self):
        """Produce a randomly picked 30Sec video for the given percentile."""
        print("****************Started producing video...**************")
        percentile = self.get_percentile()
        cap = cv2.VideoCapture(self.get_source_path())
        total_frames = self.find_total_frames(cap, self.VIDEO_DURATION_SEC)
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        whole_video_duration_sec = self.find_whole_video_duration(cap)
        start_frame = self.find_start_frame(
            percentile, total_src_frames, whole_video_duration_sec
        )
        end_frame = start_frame + total_frames
        print("Aragon says", total_frames, start_frame, end_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_list = self.find_frame_list(cap, start_frame, end_frame)
        self.write_to_mp4(cap, f"{percentile}_resampled_vid", frame_list)
        cap.release()
        cv2.destroyAllWindows()
        del cap
        gc.collect()
        print("*****************The process is completed...******************")
