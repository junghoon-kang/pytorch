import os
import glob
import pytest
from pytorch_lightning import Trainer

import config
from tests.integration.fixture import *


@pytest.fixture
def ckpt_path(cla_model, cla_dataloaders):
    ckpt_paths = glob.glob(os.path.join("checkpoints", "*.ckpt"))
    if len(ckpt_paths) > 0:
        return ckpt_paths[0]

    train_dataloader, valid_dataloader, test_dataloader = cla_dataloaders
    trainer = Trainer(
        logger=None,
        checkpoint_callback=True,
        gpus=1,
        max_epochs=10,
        min_epochs=1
    )
    trainer.fit(cla_model, train_dataloader, valid_dataloader)
    return trainer.checkpoint_callback.best_model_path
