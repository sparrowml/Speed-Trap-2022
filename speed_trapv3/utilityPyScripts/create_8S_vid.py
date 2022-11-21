from speed_trapv3.sampling import ResampleVideoToVideo as v2v

trimmed_vid = v2v(
    _src="/code/data/datasets/source_videos/25_resampled_vid.mp4",
    _dst="/code/data/datasets/source_videos",
    _percentile=20,
    _duration_in_sec=8,
)
trimmed_vid.produce_resampled_video()
