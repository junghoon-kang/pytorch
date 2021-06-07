import os, sys
import pytest

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
import config
from vision.annotation import *


@pytest.fixture
def dataset_path():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_filepath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%", "test.txt")
    seg_label_dirpath = os.path.join(path, "mask", "original.2class")
    return [image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath]

def test_ImageAnnotationSingleton(dataset_path):
    anno = SingleImageAnnotation(num_classes=2)
    anno.from_research_format(*dataset_path)
    singleton = anno[0]
    tup = anno[0].data
    assert isinstance(tup, tuple)
    assert isinstance(tup[0], str)
    assert isinstance(tup[1], int)
    assert isinstance(tup[2], str)

def test_SingleImageAnnotation_from_research_format(dataset_path):
    anno = SingleImageAnnotation(num_classes=2)
    anno.from_research_format(*dataset_path)
    assert len(anno) == 575
    assert isinstance(anno[0].image, str)
    assert isinstance(anno[0].cla_label, int)
    assert isinstance(anno[0].seg_label, str)

def test_SingleImageAnnotation_from_directory_format():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original.directory")
    image_dirpaths = [ os.path.join(path, "image", c) for c in ["OK", "NG"] ]
    cla_labels = [0, 1]
    seg_label_dirpath = os.path.join(path, "mask")

    anno = SingleImageAnnotation(num_classes=2)
    anno.from_directory_format(image_dirpaths, cla_labels, seg_label_dirpath)
    assert len(anno) == 575
    assert isinstance(anno[0].image, str)
    assert isinstance(anno[0].cla_label, int)
    assert isinstance(anno[0].seg_label, str)


if __name__ == "__main__":
    from IPython import embed; embed(); assert False
