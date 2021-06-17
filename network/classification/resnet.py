import torch
import torch.nn as nn
from torchvision.models import resnet


__all__ = [
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet101',
    'ResNet152',
]


def ResNet18(num_classes=2, pretrained=True):
    net = resnet.resnet18(pretrained=pretrained)
    # Note that resnet18(pretrained=True) and resnet18(pretrained=False) 
    # not only have different weights but also different module names.
    # Thus, you cannot checkpoint a pretrained model then load into non-pretrained model.
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def ResNet34(num_classes=2, pretrained=True):
    net = resnet.resnet34(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def ResNet50(num_classes=2, pretrained=True):
    net = resnet.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def ResNet101(num_classes=2, pretrained=True):
    net = resnet.resnet101(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

def ResNet152(num_classes=2, pretrained=True):
    net = resnet.resnet152(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

if __name__ == '__main__':
    def ResNet_test():
        net = ResNet18(num_classes=2)
        print('num_classes =', net.fc.out_features)
    #ResNet_test()
