import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


__all__ = [
    'EfficientNetB0',
    'EfficientNetB5',
]


def EfficientNetB0(num_classes=2, pretrained=True):
    net = EfficientNet.from_pretrained('efficientnet-b0')
    net._fc = nn.Linear(net._fc.in_features, num_classes)
    return net

def EfficientNetB5(num_classes=2, pretrained=True):
    net = EfficientNet.from_pretrained('efficientnet-b5')
    net._fc = nn.Linear(net._fc.in_features, num_classes)
    return net


if __name__ == '__main__':
    def EfficientNet_test():
        net = EfficientNetB5(num_classes=2)
        print('num_classes =', net._fc.out_features)
    #EfficientNet_test()
