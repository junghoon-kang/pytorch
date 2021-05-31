import os
import cv2
import skimage.io
import numpy as np
import torch
from pytorch_lightning import LightningModule


__all__ = [
    "SingleImageGradCAM",
]


class GradCAM(LightningModule):
    def __init__(self, network, layer_name="layer4.1.bn2", image_checkpoint_dir="image_checkpoint"):
        super().__init__()
        self.network = network
        self.layer_name = layer_name
        self.activation = None
        self.gradient = None
        self.register_hook()
        self.automatic_optimization = False

        for dirname in ["image", "cam", "comb"]:
            if not os.path.exists(os.path.join(image_checkpoint_dir, dirname)):
                os.makedirs(os.path.join(image_checkpoint_dir, dirname))
        self.image_checkpoint_dir = image_checkpoint_dir

    def register_hook(self):
        for name, layer in self.network.named_modules():
            if name == self.layer_name:
                layer.register_forward_hook(self.register_forward_hook)
                layer.register_backward_hook(self.register_backward_hook)
                return
        assert False

    def register_forward_hook(self, module, x, y):
        self.activation = y

    def register_backward_hook(self, module, x, y):
        self.gradient = y

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.eval()
        x, y, name = batch  # x: 16x3x120x200, y: 16
        y_ = self(x)  # y_: 16x2
        activation = self.activation  # 16x512x4x7
        H, W = x.shape[2:]

        one_hot = np.zeros(tuple(y_.size()), dtype=np.float32)
        for i in range(len(y)):
            one_hot[i][y[i]] = 1
        one_hot = torch.from_numpy(one_hot).cuda()
        one_hot.requires_grad = True
        one_hot = torch.sum(one_hot * y_)

        self.zero_grad()
        self.manual_backward(one_hot)
        grads_val = self.gradient[0].cpu().numpy()  # 16x512x4x7
        activation = activation.detach().cpu().numpy()  # 16x512x4x7
        argmax_y_ = torch.argmax(y_, dim=1).cpu().numpy()  # 16
        for i in range(len(y)):
            weights = np.mean(grads_val[i], axis=(1,2))  # 512
            cam = np.zeros(activation[i].shape[1:], dtype=np.float32)  # 4x7
            for j, w in enumerate(weights):
                cam += w * activation[i][j,:,:]
            cam = np.maximum(cam, 0)  # ReLU
            normalized_cam = cam - np.min(cam)
            normalized_cam = normalized_cam / np.max(normalized_cam)
            resized_cam = cv2.resize(normalized_cam, (W,H), interpolation=cv2.INTER_LINEAR)
            self.checkpoint_image(x[i], y[i], resized_cam, name[i])

    def get_color_map(self, x, normalize=False):
        x = np.float32(x)
        if normalize and np.max(x) != 0: x /= np.max(x)
        x = cv2.applyColorMap(np.uint8(255*x), cv2.COLORMAP_JET)
        x = np.ascontiguousarray(x[...,[2,1,0]])
        x = np.float32(x) / 255.
        return x

    def configure_optimizers(self):
        return None


class SingleImageGradCAM(GradCAM):
    def __init__(self, network, layer_name="network.layer4.1.bn2", image_checkpoint_dir="image_checkpoint"):
        super().__init__(network, layer_name, image_checkpoint_dir)

    def checkpoint_image(self, x, y, resized_cam, name):
        if y != 0:
            x = x.cpu().numpy().astype(np.float32) / 255.
            image = np.transpose(x, (1,2,0))
            if image.shape[-1] == 1:
                image = np.tile(image, 3)
            cam = self.get_color_map(resized_cam, normalize=False)
            comb = cam + image
            comb /= np.max(comb)
            image = (image * 255).astype(np.uint8)
            resized_cam = (resized_cam * 255).astype(np.uint8)
            comb = (comb * 255).astype(np.uint8)
            if name.split(".")[-1] != "jpg":
                name = ".".join(name.split(".")[:-1]) + ".jpg"
            skimage.io.imsave(
                os.path.join(self.image_checkpoint_dir, "image", name),
                image
            )
            skimage.io.imsave(
                os.path.join(self.image_checkpoint_dir, "cam", name),
                resized_cam
            )
            skimage.io.imsave(
                os.path.join(self.image_checkpoint_dir, "comb", name),
                comb
            )
