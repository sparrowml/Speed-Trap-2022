from speed_trapv3.keypoints.dataset import version_annotations

src = "/root/.darwin/datasets/sparrow-computing/kj_speedtrap/releases/allv1/annotations"
dst = "/code/data/datasets/annotations"
version_annotations(src, dst)
