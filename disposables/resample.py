from tqdm import tqdm

from speed_trapv3.sampling import *

cap = get_cap()

for _ in tqdm(range(500)):
    sample_video_frames(cap)
