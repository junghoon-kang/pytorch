import os
import pytest

import config
from vision.annotation import *


# dataset paths
@pytest.fixture
def dataset_paths():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original.rescaled256")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_dirpath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%")
    seg_label_dirpath = os.path.join(path, "mask", "labeler.2class")
    return (image_dirpath, annotation_filepath, imageset_dirpath, seg_label_dirpath)

@pytest.fixture
def trainset_paths(dataset_paths):
    (image_dirpath, annotation_filepath, imageset_dirpath, seg_label_dirpath) = dataset_paths
    return (
        image_dirpath,
        annotation_filepath,
        os.path.join(imageset_dirpath, "train.1.txt"),
        seg_label_dirpath
    )

@pytest.fixture
def validset_paths(dataset_paths):
    (image_dirpath, annotation_filepath, imageset_dirpath, seg_label_dirpath) = dataset_paths
    return (
        image_dirpath,
        annotation_filepath,
        os.path.join(imageset_dirpath, "validation.1.txt"),
        seg_label_dirpath
    )

@pytest.fixture
def testset_paths(dataset_paths):
    (image_dirpath, annotation_filepath, imageset_dirpath, seg_label_dirpath) = dataset_paths
    return (
        image_dirpath,
        annotation_filepath,
        os.path.join(imageset_dirpath, "test.txt"),
        seg_label_dirpath
    )

# annotations
@pytest.fixture
def trainset_annotation(trainset_paths):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*trainset_paths)
    return annotation

@pytest.fixture
def validset_annotation(validset_paths):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*validset_paths)
    return annotation

@pytest.fixture
def testset_annotation(testset_paths):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*testset_paths)
    return annotation

@pytest.fixture
def annotations(trainset_annotation, validset_annotation, testset_annotation):
    return (trainset_annotation, validset_annotation, testset_annotation)
