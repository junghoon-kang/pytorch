import torch.nn as nn


__all__ = [
    'CrossEntropyLoss',
    'CrossEntropyLoss2d',
]


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__(weight=weight, reduction='sum')

class CrossEntropyLoss2d(nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__(weight=weight, reduction='sum')

    def forward(self, input, target):
        N, C, H, W = input.size()
        pred = input.transpose(1,2).transpose(2,3).contiguous().view(-1, C)
        true = target.transpose(1,2).transpose(2,3).long().contiguous().flatten()
        return super(CrossEntropyLoss2d, self).forward(pred, true)
