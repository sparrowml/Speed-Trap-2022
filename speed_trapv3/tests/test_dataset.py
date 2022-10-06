import os

from speed_trapv3.keypoints.dataset import get_sample_dicts, version_annotations
from speed_trapv3.keypoints.utils import Holdout, get_holdout


def test_version_annotations_which_generates_JSON_files():
    src = "/root/.darwin/datasets/sparrow-computing/kj_speedtrap/releases/allv1/annotations"
    dst = "/code/data/datasets/annotations"
    assert len(os.listdir(dst)) == 0
    version_annotations(src, dst)
    assert len(os.listdir(dst)) > 0


def test_get_sample_dicts():
    get_sample_dicts(Holdout.TRAIN)
