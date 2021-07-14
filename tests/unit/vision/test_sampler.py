import os, sys
import pytest

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config
from vision.annotation import SingleImageAnnotation
from vision.sampler import WeightedSampler


@pytest.fixture
def dataset_path():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_filepath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%", "test.txt")
    seg_label_dirpath = os.path.join(path, "mask", "original.2class")
    return [image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath]

@pytest.fixture
def annotation(dataset_path):
    _annotation = SingleImageAnnotation(num_classes=2)
    _annotation.from_research_format(*dataset_path)
    return _annotation

def test_WeightedSampler(annotation):
    # create sampler
    sampler = WeightedSampler(annotation, weights=[3,1])

    # sanity check dataset
    assert len(sampler.subsets) == 2
    assert len(sampler.subsets[0]) == 504
    assert len(sampler.subsets[1]) == 71

    # check sampler's subset size
    assert len(sampler.subsets[0]) == 504
    assert len(sampler.subsets[1]) == 71

    # check sampler's weighted subset size
    assert sampler.weighted_subset_sizes[0] == 504
    assert sampler.weighted_subset_sizes[1] == 168

    # check sampler's total size
    assert len(sampler) == sum(sampler.weighted_subset_sizes)
