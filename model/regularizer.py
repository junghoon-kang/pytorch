import torch
import copy
from abc import ABCMeta, abstractmethod


__all__ = [
    "L2",
    "L2SP",
    "L2FM",
    "DELTA",
]

class Regularizer(metaclass=ABCMeta):
    @abstractmethod
    def calculate_loss(self, x=None):
        pass

class L2(Regularizer):
    def __init__(self, model, weight=0.1):
        self.model = model
        self.weight = 0.1

    def calculate_loss(self):
        loss = 0
        for p in tuple(self.model.parameters()):
            loss += torch.norm(p, 2) / 2
        return self.weight * loss

class L2SP(Regularizer):
    def __init__(self, model_src, model_tgt, weight=0.1):
        self.model_src = copy.deepcopy(model_src)
        self.model_tgt = model_tgt
        self.weight = weight

    def calculate_loss(self):
        loss = 0
        for ps, pt in tuple(zip(self.model_src.parameters(), self.model_tgt.parameters()))[:-2]:
            loss += torch.norm(pt - ps, 2) / 2
        return self.weight * loss

class L2FM(Regularizer):
    def __init__(
        self, model_src, model_tgt, weight=0.1,
        layers_to_be_hooked=[
            "features.4.1.conv2",
            "features.5.1.conv2",
            "features.6.1.conv2",
            "features.7.1.conv2",
        ]
    ):
        self.model_src = copy.deepcopy(model_src)
        self.model_tgt = model_tgt
        self.weight = weight
        self.layers_to_be_hooked = layers_to_be_hooked
        self.feature_maps_src = []
        self.feature_maps_tgt = []
        self.__register_hook(self.model_src, self.feature_maps_src)
        self.__register_hook(self.model_tgt, self.feature_maps_tgt)

    def __register_hook(self, model, feature_maps):
        for name, layer in model.named_modules():
            if name in self.layers_to_be_hooked:
                print(name)
                layer.register_forward_hook(lambda module, x, y: feature_maps.append(y))

    def calculate_loss(self, x):
        _ = self.model_src(x)
        loss = 0
        for fms, fmt in zip(self.feature_maps_src, self.feature_maps_tgt):
            loss += torch.norm(fmt - fms.detach(), 2) / 2
        self.feature_maps_src.clear()
        self.feature_maps_tgt.clear()
        return self.weight * loss


if __name__ == "__main__":
    import torch
    import numpy as np

    from network.resnet import *

    net = ResNet18(num_classes=2)

    def L2_test():
        reg = L2(net, weight=0.1)
        print(reg.calculate_loss())
    #L2_test()

    def L2SP_test():
        reg = L2SP(net, net, weight=0.1)
        print(reg.calculate_loss())  # tensor(0., grad_fn=<MulBackward0>)
    #L2SP_test()

    def L2FM_test():
        reg = L2FM(
            net, net, weight=0.1,
            layers_to_be_hooked=[
                "layer1.1.conv2",
                "layer2.1.conv2",
                "layer3.1.conv2",
                "layer4.1.conv2",
            ]
        )
        x = torch.Tensor( np.zeros((1,3,224,224), dtype=np.float32) )
        print(reg.calculate_loss(x))  # 0.0
    #L2FM_test()
