import os, sys
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config
from vision.annotation import *
from vision.dataset import *
from vision.transform import *


@pytest.fixture
def dataset_path():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_filepath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%", "test.txt")
    seg_label_dirpath = os.path.join(path, "mask", "original.2class")
    return [image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath]

def test_ClassificationDataset_1(dataset_path):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = ClassificationDataset(
        annotation,
        transforms=[],
        one_hot=False
    )
    assert len(dataset) > 0
    x, y, name = dataset[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (512,512)
    assert x.dtype == np.uint8
    assert isinstance(y, int)
    assert isinstance(name, str)

def test_ClassificationDataset_2(dataset_path):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = ClassificationDataset(
        annotation,
        transforms=[ToTensor()],
        one_hot=False
    )
    assert len(dataset) > 0
    x, y, name = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([1,512,512])
    assert x.dtype == torch.float32
    assert isinstance(y, int)
    assert isinstance(name, str)

def test_ClassificationDataset_3(dataset_path):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = ClassificationDataset(
        annotation,
        transforms=[ToTensor()],
        one_hot=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    for x, y, name in data_loader:
        assert isinstance(x, torch.Tensor)
        assert x.shape == torch.Size([1,1,512,512])
        assert x.dtype == torch.float32
        assert isinstance(y, torch.Tensor)
        assert y.shape == torch.Size([1])
        assert y.dtype == torch.int64
        assert isinstance(name, list)
        assert len(name) == 1
        assert isinstance(name[0], str)
        break

@pytest.mark.parametrize("batch_size", [4,8])
def test_ClassificationDataset_4(dataset_path, batch_size):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = ClassificationDataset(
        annotation,
        transforms=[ToTensor()],
        one_hot=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    for x, y, name in data_loader:
        assert x.shape == torch.Size([batch_size,1,512,512])
        assert y.shape == torch.Size([batch_size])
        assert len(name) == batch_size
        break

#######################
# SegmentationDataset #
#######################

def test_SegmentationDataset_1(dataset_path):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = SegmentationDataset(
        annotation,
        transforms=[],
        one_hot=False
    )
    assert len(dataset) > 0
    x, y, name = dataset[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (512,512)
    assert x.dtype == np.uint8
    assert isinstance(y, np.ndarray)
    assert y.shape == (512,512)
    assert x.dtype == np.uint8
    assert isinstance(name, str)

def test_SegmentationDataset_2(dataset_path):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = SegmentationDataset(
        annotation,
        transforms=[ToTensor()],
        one_hot=False
    )
    assert len(dataset) > 0
    x, y, name = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([1,512,512])
    assert x.dtype == torch.float32
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([512,512])
    assert y.dtype == torch.uint8
    assert isinstance(name, str)

def test_SegmentationDataset_3(dataset_path):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = SegmentationDataset(
        annotation,
        transforms=[ToTensor()],
        one_hot=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    for x, y, name in data_loader:
        assert isinstance(x, torch.Tensor)
        assert x.shape == torch.Size([1,1,512,512])
        assert x.dtype == torch.float32
        assert isinstance(y, torch.Tensor)
        assert y.shape == torch.Size([1,512,512])
        assert y.dtype == torch.uint8
        assert isinstance(name, list)
        assert len(name) == 1
        assert isinstance(name[0], str)
        break

@pytest.mark.parametrize("batch_size", [4,8])
def test_SegmentationDataset_4(dataset_path, batch_size):
    annotation = SingleImageAnnotation(num_classes=2)
    annotation.from_research_format(*dataset_path)
    dataset = SegmentationDataset(
        annotation,
        transforms=[ToTensor()],
        one_hot=False
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    for x, y, name in data_loader:
        assert x.shape == torch.Size([batch_size,1,512,512])
        assert y.shape == torch.Size([batch_size,512,512])
        assert len(name) == batch_size
        break


if __name__ == "__main__":
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_filepath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%", "test.txt")
    seg_label_dirpath = os.path.join(path, "mask", "original.2class")


    anno = SingleImageAnnotation(num_classes=2)
    anno.from_research_format(image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath)
    dataset = ClassificationDataset(anno, transforms=[])
    from IPython import embed; embed(); assert False
