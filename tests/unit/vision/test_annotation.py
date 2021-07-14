import os, sys
import pytest

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config
from vision.annotation import *
from tests.fixture import (
    dataset_paths,
    testset_paths
)


def test_ImageAnnotationSingleton(testset_paths):
    anno = SingleImageAnnotation(num_classes=2)
    anno.from_research_format(*testset_paths)
    singleton = anno[0]
    tup = anno[0].data
    assert isinstance(tup, tuple)
    assert isinstance(tup[0], str)
    assert isinstance(tup[1], int)
    assert isinstance(tup[2], str)

def test_SingleImageAnnotation_from_research_format(testset_paths):
    anno = SingleImageAnnotation(num_classes=2)
    anno.from_research_format(*testset_paths)
    assert len(anno) == 575
    assert isinstance(anno[0].image, str)
    assert isinstance(anno[0].cla_label, int)
    assert isinstance(anno[0].seg_label, str)

def test_SingleImageAnnotation_from_directory_format():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original.directory", "test")
    image_dirpaths = [ os.path.join(path, "image", c) for c in ["OK", "NG"] ]
    cla_labels = [0, 1]
    seg_label_dirpath = os.path.join(path, "mask")

    anno = SingleImageAnnotation(num_classes=2)
    anno.from_directory_format(image_dirpaths, cla_labels, seg_label_dirpath)
    assert len(anno) == 575
    assert isinstance(anno[0].image, str)
    assert isinstance(anno[0].cla_label, int)
    assert isinstance(anno[0].seg_label, str)
