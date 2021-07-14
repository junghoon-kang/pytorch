import os, sys
import glob
import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config
from callback.checkpointer import ModelCheckpoint
from tests.integration.fixture import *


def test_train(seg_model, seg_dataloaders):
    train_dataloader, valid_dataloader, test_dataloader = seg_dataloaders

    logger = TensorBoardLogger(save_dir=os.path.join(PATH, "checkpoint"), name=None, version="segmentation")

    def is_better(new, old):
        smaller_loss = new["valid_loss"] < old["valid_loss"]
        return ( new["valid_IoU"] > old["valid_IoU"] ) or \
            ( new["valid_IoU"] == old["valid_IoU"] and smaller_loss )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.root_dir, logger.version),
        filename="{epoch:03d}-{valid_IoU:.4f}",
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
            checkpoint_callback,
        ],
        gpus=1,
        max_epochs=10,
        min_epochs=1
    )
    trainer.fit(seg_model, train_dataloader, valid_dataloader)

    best_score = checkpoint_callback.best_model_score
    assert best_score["valid_IoU"] > 0.5, best_score


def test_evaluate(seg_model, seg_dataloaders):
    train_dataloader, valid_dataloader, test_dataloader = seg_dataloaders

    ckpt_paths = sorted(glob.glob(os.path.join(PATH, "checkpoint", "segmentation", "*.ckpt")))
    ckpt = torch.load(ckpt_paths[0])
    seg_model.load_state_dict(ckpt["state_dict"])

    trainer = Trainer(
        logger=False,
        callbacks=None,
        gpus=1
    )
    result = trainer.test(seg_model, test_dataloaders=test_dataloader)
    assert result[0]["test_IoU"] > 0.5, result
