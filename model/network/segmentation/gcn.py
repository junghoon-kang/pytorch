import torch
import torch.nn as nn

from model.network.classification.resnet import *


__all__ = [
    'GCResNet18',
    'GCResNet50',
]

class BRM(nn.Module):
    def __init__(self, out_channel):
        super(BRM, self).__init__()
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return x + out


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel, k):
        super(GCM, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channel, out_channel, kernel_size=(k,1), padding=(k//2,0), bias=False)
        self.conv_l2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1,k), padding=(0,k//2), bias=False)
        self.conv_r1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,k), padding=(k//2,0), bias=False)
        self.conv_r2 = nn.Conv2d(out_channel, out_channel, kernel_size=(k,1), padding=(0,k//2), bias=False)

    def forward(self, x):
        l = self.conv_l1(x)
        l = self.conv_l2(l)
        r = self.conv_r1(x)
        r = self.conv_r2(r)
        return l + r


class GCResNet(nn.Module):
    def __init__(self, resnet, out_channel=30, kernel_size=7, num_classes=2):
        super(GCResNet, self).__init__()
        self.resnet = resnet
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.num_classes = num_classes

        self.gcm1 = GCM(self.resnet.fc.in_features, out_channel, kernel_size)
        self.brm1 = BRM(out_channel)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.gcm2 = GCM(self.resnet.fc.in_features//2, out_channel, kernel_size)
        self.brm2_1 = BRM(out_channel)
        self.brm2_2 = BRM(out_channel)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.gcm3 = GCM(self.resnet.fc.in_features//4, out_channel, kernel_size)
        self.brm3_1 = BRM(out_channel)
        self.brm3_2 = BRM(out_channel)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.gcm4 = GCM(self.resnet.fc.in_features//8, out_channel, kernel_size)
        self.brm4_1 = BRM(out_channel)
        self.brm4_2 = BRM(out_channel)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.brm5_1 = BRM(out_channel)
        self.brm5_2 = BRM(out_channel)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(out_channel, num_classes, 1, bias=True)

    def forward(self, x):
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)  # max pool
        layer1 = self.resnet.layer1(out)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        gclayer1 = self.upsample1(self.brm1(self.gcm1(layer4)))
        gclayer2 = self.upsample2(self.brm2_2(self.brm2_1(self.gcm2(layer3)) + gclayer1))
        gclayer3 = self.upsample3(self.brm3_2(self.brm3_1(self.gcm3(layer2)) + gclayer2))
        gclayer4 = self.upsample4(self.brm4_2(self.brm4_1(self.gcm4(layer1)) + gclayer3))
        out = self.brm5_2(self.upsample5(self.brm5_1(gclayer4)))
        out = self.conv(out)
        return out


def GCResNet18(num_classes=2, out_channel=30, kernel_size=7):
    resnet = ResNet18(num_classes, pretrained=True)
    network = GCResNet(resnet, out_channel=out_channel, kernel_size=kernel_size, num_classes=num_classes)
    return network


def GCResNet50(num_classes=2, out_channel=30, kernel_size=7):
    resnet = ResNet50(num_classes, pretrained=True)
    network = GCResNet(resnet, out_channel=out_channel, kernel_size=kernel_size, num_classes=num_classes)
    return network
