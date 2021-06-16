import os, sys
import glob
import pytest
import torch
from torch.utils.data import DataLoader
import albumentations as A
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
import config
from vision.annotation import *
from vision.dataset import *
from vision.transform import *
from vision.sampler import *
from model.classification import *
from model.network.classification.resnet import *
from model.criterion import *
from model.optimizer import *
from model.checkpointer import *


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

    sampler = WeightedSampler([1,1])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler(train_dataset),
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
def model():
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
def logger():
    logger = TensorBoardLogger(save_dir=os.path.join(PATH, "checkpoint"), name="main")
    return logger

def test_train(model, dataloaders, logger):
    train_dataloader, valid_dataloader, test_dataloader = dataloaders

    def is_better(new, old):
        smaller_loss = new["valid_loss"] < old["valid_loss"]
        return ( new["valid_Recall"] > old["valid_Recall"] ) or \
            ( new["valid_Recall"] == old["valid_Recall"] and smaller_loss )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.root_dir, f"version_{logger.version}"),
        filename="{epoch:03d}-{valid_Recall:.4f}",
        save_last=False,
        save_top_k=1,
        is_better=is_better,
        save_weights_only=False,
        auto_insert_metric_name=True,
        every_n_train_steps=None,
        every_n_val_epochs=None,
        verbose=False,
    )

    trainer = Trainer(
        logger=logger,
        checkpoint_callback=False,
        callbacks=[
            LearningRateMonitor(),
            checkpoint_callback,
        ],
        gpus=1,
        max_epochs=10,
        min_epochs=1
    )
    trainer.fit(model, train_dataloader, valid_dataloader)

    best_score = checkpoint_callback.best_model_score
    assert best_score['train_loss'] < 0.1, best_score
    assert best_score['valid_loss'] < 0.1, best_score

def test_evaluate(model, dataloaders):
    train_dataloader, valid_dataloader, test_dataloader = dataloaders

    ckpt_path = sorted(glob.glob(os.path.join(PATH, "checkpoint", "main", "version_0", "*.ckpt")))[-1]
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    trainer = Trainer(
        logger=False,
        callbacks=None,
        gpus=1
    )
    result = trainer.test(model, test_dataloaders=test_dataloader)
    assert result[0]["test_Accuracy"] == 1, result
    assert result[0]["test_Recall"] == 1, result
