import torch.nn as nn


__all__ = [
    'CrossEntropyLoss',
    'CrossEntropyLoss2d',
]


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, temperature_scale=1.):
        super(CrossEntropyLoss, self).__init__(weight=weight, reduction='sum')
        self.t_scale = temerature_scale

    def forward(self, pred, true):
        return super(CrossEntropyLoss, self).forward(pred / self.t_scale, true)

class CrossEntropyLoss2d(nn.CrossEntropyLoss):
    def __init__(self, weight=None, temperature_scale=1.):
        super(CrossEntropyLoss2d, self).__init__(weight=weight, reduction='sum')
        self.t_scale = temerature_scale

    def forward(self, pred, true):
        N, C, H, W = pred.size()
        pred = pred.transpose(1,2).transpose(2,3).contiguous().view(-1, C)
        true = true.transpose(1,2).transpose(2,3).long().contiguous().flatten()
        return super(CrossEntropyLoss2d, self).forward(pred / self.t_scale, true)
