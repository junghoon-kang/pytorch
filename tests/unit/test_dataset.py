import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2 as ToTensor

import os, sys
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
import config
from vision.dataset import *


@pytest.fixture
def dataset_path():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_filepath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%", "test.txt")
    seg_label_dirpath = os.path.join(path, "mask", "original.2class")
    return [image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath]

def test_SingleImageClassificationDataset_1(dataset_path):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    dataset = SingleImageClassificationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        transforms=[]
    )
    assert len(dataset) > 0
    x, y, name = dataset[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (512,512)
    assert isinstance(y, int)
    assert isinstance(name, str)

def test_SingleImageClassificationDataset_2(dataset_path):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    dataset = SingleImageClassificationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        transforms=[ToTensor()]
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
        assert x.size() == torch.Size([1,1,512,512])
        assert isinstance(y, torch.Tensor)
        assert y.size() == torch.Size([1])
        assert isinstance(name, list)
        assert len(name) == 1
        assert isinstance(name[0], str)
        break

@pytest.mark.parametrize("batch_size", [4,8])
def test_SingleImageClassificationDataset_3(dataset_path, batch_size):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    dataset = SingleImageClassificationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        transforms=[ToTensor()]
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    for x, y, name in data_loader:
        assert x.size() == torch.Size([batch_size,1,512,512])
        assert y.size() == torch.Size([batch_size])
        assert len(name) == batch_size
        break

def test_SingleImageSegmentationDataset_1(dataset_path):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    dataset = SingleImageSegmentationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        seg_label_dirpath,
        transforms=[]
    )
    assert len(dataset) > 0
    x, y, name = dataset[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (512,512)
    assert isinstance(y, np.ndarray)
    assert y.shape == (512,512)
    assert isinstance(name, str)

def test_SingleImageSegmentationDataset_2(dataset_path):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    dataset = SingleImageSegmentationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        seg_label_dirpath,
        transforms=[ToTensor()]
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
        assert x.size() == torch.Size([1,1,512,512])
        assert isinstance(y, torch.Tensor)
        assert y.size() == torch.Size([1,512,512])
        assert isinstance(name, list)
        assert len(name) == 1
        assert isinstance(name[0], str)
        break

@pytest.mark.parametrize("batch_size", [4,8])
def test_SingleImageSegmentationDataset_3(dataset_path, batch_size):
    image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath = dataset_path
    dataset = SingleImageSegmentationDataset(
        image_dirpath,
        annotation_filepath,
        imageset_filepath,
        seg_label_dirpath,
        transforms=[ToTensor()]
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    for x, y, name in data_loader:
        assert x.size() == torch.Size([batch_size,1,512,512])
        assert y.size() == torch.Size([batch_size,512,512])
        assert len(name) == batch_size
        break
