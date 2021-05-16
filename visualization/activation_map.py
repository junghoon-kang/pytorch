import os
import cv2
import shutil
import skimage.io
import skimage.transform
import numpy as np
import torch
import torch.utils.data
import torch.autograd
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ['ActivationMap']


class ActivationMap:
    def __init__(self, model, activation_layers, args):
        self.model = model
        self.activation_layers = activation_layers
        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.register_hook()

    def register_hook(self):
        for name, layer in self.model.named_modules():
            if name in self.activation_layers:
                layer.register_forward_hook(self.append_activation(name))

    def append_activation(self, name):
        def _append_activation(module, x, y):
            self.activations[name].append(y)
        return _append_activation

    def setup(self):
        self.activations = { layer: [] for layer in self.activation_layers }

    def checkpoint(self, dataset, activation_checkpoint_dir='activation_checkpoint'):
        for activation_layer in self.activation_layers:
            if not os.path.exists(f'{activation_checkpoint_dir}/{activation_layer}'):
                os.makedirs(f'{activation_checkpoint_dir}/{activation_layer}')

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.model.eval()
        for x, y, name in data_loader:
            self.setup()
            x = x.cuda(self.gpu)  # x: 16x3x512x512
            y_ = self.model(x)  # y_: 16x2
            for layer in self.activation_layers:
                for i in range(len(name)):
                    activation = self.activations[layer][0][i]
                    image_name = '.'.join(name[i].split('.')[:-1])
                    np.save(f'{activation_checkpoint_dir}/{layer}/{image_name}.npy', activation.cpu().detach().numpy())
