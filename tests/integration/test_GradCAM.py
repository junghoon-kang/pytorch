import os, sys
import glob
import pytest
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
import config
from vision.dataset import *
from vision.transform import *
from vision.sampler import *
from model.network.classification.resnet import *
from visualization.grad_cam import *


path = os.path.join(config.DATA_DIR, "public", "DAGM", "original.rescaled256")
image_dirpath = os.path.join(path, "image")
annotation_filepath = os.path.join(path, "annotation", "domain1.single_image.2class.json")
imageset_dirpath = os.path.join(path, "imageset", "domain1.single_image.2class", "public", "ratio", "100%")
seg_label_dirpath = os.path.join(path, "mask", "original.2class")
train_filepath = os.path.join(imageset_dirpath, "train.1.txt")
valid_filepath = os.path.join(imageset_dirpath, "validation.1.txt")
test_filepath  = os.path.join(imageset_dirpath, "test.txt")

train_dataset = SingleImageClassificationDataset(
    image_dirpath, annotation_filepath, train_filepath, seg_label_dirpath,
    transforms=[
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
test_dataset = SingleImageClassificationDataset(
    image_dirpath, annotation_filepath, test_filepath, seg_label_dirpath,
    transforms=[
        To3channel(),
        ToTensor(),
    ]
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
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
    num_workers=1,
    pin_memory=True
)

network = ResNet18(num_classes=2).cuda(0)
model = SingleImageGradCAM(network)

ckpt = torch.load(os.path.join(PATH, "checkpoint", "main", "version_0", "last.ckpt"))
model.load_state_dict(ckpt["state_dict"])

trainer = Trainer(
    logger=False,
    checkpoint_callback=False,
    gpus=1,
    max_epochs=1
)
trainer.fit(model, test_dataloader)
