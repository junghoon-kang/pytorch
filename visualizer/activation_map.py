import os
import numpy as np
from pytorch_lightning import LightningModule


__all__ = [
    "ActivationMap",
]


class ActivationMap(LightningModule):
    def __init__(self, network, layer_names=[], checkpoint_dir="activation_checkpoint"):
        super().__init__()
        self.network = network
        self.layer_names = layer_names
        self.register_hook()
        self.automatic_optimization = False

        for dirname in layer_names:
            if not os.path.exists(os.path.join(checkpoint_dir, dirname)):
                os.makedirs(os.path.join(checkpoint_dir, dirname))
        self.checkpoint_dir = checkpoint_dir

    def register_hook(self):
        for name, layer in self.network.named_modules():
            if name in self.layer_names:
                layer.register_forward_hook(self.append_activation(name))

    def append_activation(self, name):
        def _append_activation(module, x, y):
            self.activations[name].append(y)
        return _append_activation

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.clear()
        self.eval()
        x, y, name = batch  # x: 16x3x120x200, y: 16
        y_ = self(x)  # y_: 16x2
        for layer in self.layer_names:
            for i in range(len(name)):
                activation = self.activations[layer][0][i]
                image_name = ".".join(name[i].split(".")[:-1]) + ".npy"
                np.save(
                    os.path.join(self.checkpoint_dir, layer, image_name),
                    activation.cpu().detach().numpy()
                )

    def clear(self):
        self.activations = { layer: [] for layer in self.layer_names }

    def configure_optimizers(self):
        return None
