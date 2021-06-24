import torch.nn.functional as F
from torchmetrics import MetricCollection
from pytorch_lightning import LightningModule


__all__ = [
    "Classification"
]


class Classification(LightningModule):
    def __init__(self, network, criterion, optimizer, scheduler=None, regularizer=None, metrics=[], **kwargs):
        super().__init__()

        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer
        self.metrics = MetricCollection(metrics)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_ = self(x)
        loss = self.criterion(y_, y)
        if self.regularizer is not None:
            loss += self.regularizer.calculate_loss()
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_ = self(x)
        loss = self.criterion(y_, y)
        prob = F.softmax(y_, dim=1)
        self.metrics.update(prob, y)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        results = self.metrics.compute()
        for k, v in sorted(results.items()):
            self.log(f"valid_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_ = self(x)
        loss = self.criterion(y_, y)
        prob = F.softmax(y_, dim=1)
        self.metrics.update(prob, y)

    def test_epoch_end(self, outputs):
        results = self.metrics.compute()
        for k, v in sorted(results.items()):
            self.log(f"test_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer
        else:
            return [self.optimizer], [self.scheduler]
