import os
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import config
from vision.transform import *
from vision.dataset import *
from vision.sampler import *
from model.classification import *
from model.network.classification.resnet import *
from model.criterion import *
from model.optimizer import *
from model.scheduler import *
from model.regularizer import *
from model.metric import *


path = os.path.join(config.DATA_DIR, "public", "DAGM", "original.rescaled256")
image_dirpath = os.path.join(path, "image")
annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
imageset_dirpath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%")
seg_label_dirpath = os.path.join(path, "mask", "original.2class")
train_filepath = os.path.join(imageset_dirpath, "train.1.txt")
valid_filepath = os.path.join(imageset_dirpath, "validation.1.txt")
test_filepath  = os.path.join(imageset_dirpath, "test.txt")

def train():
    train_dataset = SingleImageClassificationDataset(
        image_dirpath, annotation_filepath, train_filepath, seg_label_dirpath,
        transforms=[
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5),
            To3channel(),
            ToTensor(),
        ]
    )
    valid_dataset = SingleImageClassificationDataset(
        image_dirpath, annotation_filepath, valid_filepath, seg_label_dirpath,
        transforms=[
            To3channel(),
            ToTensor(),
        ]
    )

    sampler = WeightedSampler([1,1])

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler(train_dataset),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    network = ResNet18(num_classes=2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(network.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = WarmupLR(
        optimizer,
        warmup_iterations=1,
        next_scheduler=CosineAnnealingWarmRestarts(optimizer, T_0=10)
    )
    regularizer = L2(network, weight=0.1)

    model = Classification(
        network, criterion, optimizer,
        scheduler=scheduler,
        regularizer=regularizer,
        metrics=[
            torchmetrics.Accuracy(),
            torchmetrics.Recall(average="macro", num_classes=2),
        ]
    )
    checkpoint_callback = ModelCheckpoint(
        #save_top_k=1,
        #save_last=True,
        #monitor="val_Recall",
        #mode="max",
        dirpath="./logging/foo"
        #filename="{epoch:03d}-{val_Recall:.4f}",
    )
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="./logging", name="foo"),
        checkpoint_callback=checkpoint_callback,
        callbacks=[LearningRateMonitor()],
        gpus=1,
        max_epochs=10,
        min_epochs=1
    )
    trainer.fit(model, train_data_loader, valid_data_loader)
    from IPython import embed; embed(); assert False


def evaluate():
    test_dataset = SingleImageClassificationDataset(
        image_dirpath, annotation_filepath, test_filepath, seg_label_dirpath,
        transforms=[
            To3channel(),
            ToTensor(),
        ]
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    network = ResNet18(num_classes=2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(network.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = WarmupLR(
        optimizer,
        warmup_iterations=1,
        next_scheduler=CosineAnnealingWarmRestarts(optimizer, T_0=10)
    )
    regularizer = L2(network, weight=0.1)

    model = Classification(
        network, criterion, optimizer,
        scheduler=scheduler,
        regularizer=regularizer,
        metrics=[
            torchmetrics.Accuracy(),
            torchmetrics.Recall(average="macro", num_classes=2),
        ]
    )
    from IPython import embed; embed(); assert False

    #trainer = Trainer(
    #    logger=TensorBoardLogger(save_dir="./logging", name="foo"),
    #    checkpoint_callback=ModelCheckpoint(
    #        save_top_k=1,
    #        save_last=True,
    #        monitor="val_Recall",
    #        mode="max",
    #        filename="{epoch:03d}-{val_Recall:.4f}",
    #    ),
    #    callbacks=[LearningRateMonitor()],
    #    gpus=1,
    #    max_epochs=100,
    #    min_epochs=1
    #)
    #trainer.fit(model, train_data_loader, valid_data_loader)

if __name__ == "__main__":
    train()
    #evaluate()
