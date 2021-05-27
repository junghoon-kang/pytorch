import os, sys
import pytest
import numpy as np
import albumentations as A
from skimage.io import imread

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
import config
from vision.transform import *


@pytest.fixture
def sample_names():
    return [
        "domain1.test.NG.0002.png",
        "domain1.train.NG.0616.png",
        "domain2.test.NG.0003.png",
        "domain3.test.NG.0007.png",
        "domain4.test.NG.0022.png",
        "domain5.test.NG.0001.png",
        "domain6.test.NG.0021.png",
        "domain7.test.NG.0005.png",
        "domain8.test.NG.0007.png",
        "domain9.test.NG.0009.png",
        "domain10.test.NG.0013.png"
    ]

@pytest.fixture
def samples(sample_names):
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    images = [ imread(os.path.join(path, "image", name)) for name in sample_names ]
    labels = [ imread(os.path.join(path, "mask", "labeler.2class", name)) for name in sample_names ]
    return zip(images, labels, sample_names)

@pytest.fixture
def RandomCropNearDefect_answers(sample_names):
    answer_dirpath = os.path.join(PATH, "test_transform", "RandomCropNearDefect")
    images = [ imread(os.path.join(answer_dirpath, "image", name)) for name in sample_names ]
    labels = [ imread(os.path.join(answer_dirpath, "label", name)) for name in sample_names ]
    return zip(images, labels, sample_names)

def test_RandomCropNearDefect(samples, RandomCropNearDefect_answers):
    for (image, label, name), (answer_image, answer_label, answer_name) in zip(samples,RandomCropNearDefect_answers):
        result = A.Compose([
            RandomCropNearDefect(
                size = (128,128),
                coverage_size = (1,1),
                fixed = True
            )
        ])(image=image, mask=label)

        output_image = result["image"]
        output_label = result["mask"]
        output_label[np.where(output_label == 0)] = 255
        output_label[np.where(output_label == 1)] = 0

        assert name == answer_name
        assert np.array_equal(output_image, answer_image)
        assert np.array_equal(output_label, answer_label)

def test_To3channel(samples):
    for image, label, name in samples:
        result = A.Compose([
            To3channel(),
        ])(image=image, mask=label)
        output_image = result["image"]
        output_label = result["mask"]
        assert output_image.shape == (512,512,3)
        assert output_label.shape == (512,512)
