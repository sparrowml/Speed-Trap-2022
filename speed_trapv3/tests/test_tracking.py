from speed_trapv3.tracking.tracking import track_objects

src_video_path = "/code/data/datasets/common_hall/source_video/25_resampled_vid.mp4"
model_path = "/code/data/models/detection/model.onnx"
track_objects(src_video_path, model_path)
