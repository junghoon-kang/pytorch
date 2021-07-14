import os, sys
import pytest

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config
from vision.sampler import WeightedSampler
from tests.fixture import (
    dataset_paths,
    testset_paths,
    testset_annotation
)


def test_WeightedSampler(testset_annotation):
    # create sampler
    sampler = WeightedSampler(testset_annotation, weights=[3,1])

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
