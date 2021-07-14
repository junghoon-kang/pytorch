import os, sys
import glob
import pytest
import torchmetrics
import albumentations as A
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config
from vision.annotation import SingleImageAnnotation
from vision.dataset import ClassificationDataset
from vision.transform import To3channel, ToTensor
from vision.sampler import WeightedSampler
from task.classification import Classification
from network.classification.resnet import ResNet18
from learner.criterion import CrossEntropyLoss
from learner.optimizer import Adam


__all__ = [
    "dataloaders",
    "cla_model",
    "ckpt_path",
]


@pytest.fixture
def dataloaders():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original.rescaled256")
    image_dirpath = os.path.join(path, "image")
    annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
    imageset_dirpath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%")
    seg_label_dirpath = os.path.join(path, "mask", "original.2class")
    train_filepath = os.path.join(imageset_dirpath, "train.1.txt")
    valid_filepath = os.path.join(imageset_dirpath, "validation.1.txt")
    test_filepath  = os.path.join(imageset_dirpath, "test.txt")

    train_annotation = SingleImageAnnotation(num_classes=2)
    valid_annotation = SingleImageAnnotation(num_classes=2)
    test_annotation = SingleImageAnnotation(num_classes=2)
    train_annotation.from_research_format(image_dirpath, annotation_filepath, train_filepath, seg_label_dirpath)
    valid_annotation.from_research_format(image_dirpath, annotation_filepath, valid_filepath, seg_label_dirpath)
    test_annotation.from_research_format(image_dirpath, annotation_filepath, test_filepath, seg_label_dirpath)

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
    return cla_model

@pytest.fixture
def ckpt_path(cla_model, dataloaders):
    ckpt_paths = glob.glob(os.path.join("checkpoints", "*.ckpt"))
    if len(ckpt_paths) > 0:
        return ckpt_paths[0]

    train_dataloader, valid_dataloader, test_dataloader = dataloaders
    trainer = Trainer(
        logger=None,
        checkpoint_callback=True,
        gpus=1,
        max_epochs=10,
        min_epochs=1
    )
    trainer.fit(cla_model, train_dataloader, valid_dataloader)
    return trainer.checkpoint_callback.best_model_path

