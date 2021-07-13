import os, sys
import pytest
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
from network.classification.resnet import ResNet18
from visualizer.grad_cam import SingleImageGradCAM
from tests.integration.visualizer.fixture import dataloaders, ckpt_path


def test_GradCam(dataloaders, ckpt_path):
    train_dataloader, valid_dataloader, test_dataloader = dataloaders

    network = ResNet18(num_classes=2)
    model = SingleImageGradCAM(
        network,
        layer_name="layer4.1.bn2",
        checkpoint_dir=os.path.join("image_checkpoint", "grad_cam")
    )

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        gpus=1,
        max_epochs=1
    )
    trainer.fit(model, test_dataloader)
