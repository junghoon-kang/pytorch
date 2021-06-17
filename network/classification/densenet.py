import torch
import torch.nn as nn
from torchvision.models import densenet


__all__ = [
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
    'DenseNet161',
]


def DenseNet121(num_classes=2, pretrained=True):
    net = densenet.densenet121(pretrained=pretrained)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net

def DenseNet169(num_classes=2, pretrained=True):
    net = densenet.densenet169(pretrained=pretrained)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net

def DenseNet201(num_classes=2, pretrained=True):
    net = densenet.densenet201(pretrained=pretrained)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net

def DenseNet161(num_classes=2, pretrained=True):
    net = densenet.densenet161(pretrained=pretrained)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net


if __name__ == '__main__':
    def DenseNet_test():
        net = DenseNet121(num_classes=2)
        print('num_classes =', net.classifier.out_features)
    #DenseNet_test()
