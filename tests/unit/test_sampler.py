import pytest
import os, sys

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
import config
from vision.dataset import *
from vision.sampler import *


@pytest.fixture
def dataset():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_filepath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%", "train.1.txt")
    return SingleImageClassificationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        transforms=[]
    )

def test_WeightedSampler(dataset):
    # sanity check dataset
    assert len(dataset.subsets) == 2
    assert len(dataset.subsets[0]) == 397
    assert len(dataset.subsets[1]) == 64

    # create sampler
    sampler = WeightedSampler(weights=[3,1])(dataset)

    # check sampler's subset size
    assert len(sampler.subsets[0]) == 397
    assert len(sampler.subsets[1]) == 64

    # check sampler's weighted subset size
    assert sampler.weighted_subset_sizes[0] == 399
    assert sampler.weighted_subset_sizes[1] == 133

    # check sampler's total size
    assert len(sampler) == sum(sampler.weighted_subset_sizes)
