import numpy as np
import torch.optim as optim
from abc import abstractmethod, abstractproperty, ABCMeta


__all__ = [
    'SGD',
    'Adam',
]


class Base(metaclass=ABCMeta):
    def get_lr(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
            return lr

    def update_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

class SGD(optim.SGD, Base):
    def __init__(self, params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

class Adam(optim.Adam, Base):
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
