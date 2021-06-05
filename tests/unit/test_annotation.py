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

def test_ImageAnnotation(dataset_path):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    anno = ImageAnnotation()
    anno.from_research_format(*dataset_path)
    assert len(anno) == 575
    assert isinstance(anno[0], ImageAnnotationSingleton)
    assert isinstance(anno[0].image, str)
    assert isinstance(anno[0].cla_label, list)
    assert isinstance(anno[0].cla_label[0], int)
    assert isinstance(anno[0].seg_label, str)
