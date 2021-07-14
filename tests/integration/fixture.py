import os
import glob
import pytest
import torchmetrics
import albumentations as A
from torch.utils.data import DataLoader

import config
from vision.dataset import ClassificationDataset, SegmentationDataset
from vision.transform import To3channel, ToTensor
from vision.sampler import WeightedSampler
from task.classification import Classification
from task.segmentation import Segmentation
from network.classification.resnet import ResNet18
from network.segmentation.gcn import GCResNet18
from learner.criterion import CrossEntropyLoss, CrossEntropyLoss2d
from learner.optimizer import Adam
from tests.fixture import *


# datasets
@pytest.fixture
def cla_datasets(annotations):
    (train_annotation, valid_annotation, test_annotation) = annotations

    train_dataset = ClassificationDataset(
        train_annotation,
        transforms=[
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5),
            To3channel(),
            ToTensor(),
        ]
    )
    valid_dataset = ClassificationDataset(
        valid_annotation,
        transforms=[
            To3channel(),
            ToTensor(),
        ]
    )
    test_dataset = ClassificationDataset(
        test_annotation,
        transforms=[
            To3channel(),
            ToTensor(),
        ]
    )
    return train_dataset, valid_dataset, test_dataset

@pytest.fixture
def seg_datasets(annotations):
    (train_annotation, valid_annotation, test_annotation) = annotations

    train_dataset = SegmentationDataset(
        train_annotation,
        transforms=[
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5),
            To3channel(),
            ToTensor(),
        ]
    )
    valid_dataset = SegmentationDataset(
        valid_annotation,
        transforms=[
            To3channel(),
            ToTensor(),
        ]
    )
    test_dataset = SegmentationDataset(
        test_annotation,
        transforms=[
            To3channel(),
            ToTensor(),
        ]
    )
    return train_dataset, valid_dataset, test_dataset


# dataloaders
@pytest.fixture
def cla_dataloaders(annotations, cla_datasets):
    (train_annotation, valid_annotation, test_annotation) = annotations
    (train_dataset, valid_dataset, test_dataset) = cla_datasets

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=WeightedSampler(train_annotation, [1,1]),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_dataloader, valid_dataloader, test_dataloader

@pytest.fixture
def seg_dataloaders(annotations, seg_datasets):
    (train_annotation, valid_annotation, test_annotation) = annotations
    (train_dataset, valid_dataset, test_dataset) = seg_datasets

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=WeightedSampler(train_annotation, [1,1]),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_dataloader, valid_dataloader, test_dataloader


# models
@pytest.fixture
def cla_model():
    network = ResNet18(num_classes=2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(network.parameters(), lr=0.0001, weight_decay=0.0001)

    model = Classification(
        network, criterion, optimizer,
        metrics=[
            torchmetrics.Accuracy(),
            torchmetrics.Recall(average="macro", num_classes=2),
        ]
    )
    return model

@pytest.fixture
def seg_model():
    network = GCResNet18(num_classes=2)
    criterion = CrossEntropyLoss2d()
    optimizer = Adam(network.parameters(), lr=0.0001, weight_decay=0.0001)

    model = Segmentation(
        network, criterion, optimizer,
        metrics=[
            torchmetrics.IoU(num_classes=2),
        ]
    )
    return model
