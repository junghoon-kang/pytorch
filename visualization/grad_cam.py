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

from vision.transform import get_cropped_image_boxes, get_combined_image, get_color_map


__all__ = ['SingleImageGradCAM']

class GradCAM(metaclass=ABCMeta):
    def checkpoint(self, dataset, image_checkpoint_dir='image_checkpoint'):
        if not os.path.exists(image_checkpoint_dir + '/image'):
            os.makedirs(image_checkpoint_dir + '/image')
        if not os.path.exists(image_checkpoint_dir + '/cam'):
            os.makedirs(image_checkpoint_dir + '/cam')
        if not os.path.exists(image_checkpoint_dir + '/comb'):
            os.makedirs(image_checkpoint_dir + '/comb')
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.model.eval()
        for x, y, name in data_loader:
            self.checkpoint_images(x, y, image_checkpoint_dir, name)

    def checkpoint_images(self, x, y, image_checkpoint_dir, name):
        x, y = x.cuda(self.gpu), y.numpy()  # x: 16x3x120x200, y: 16
        y_ = self.model(x)  # y_: 16x2
        activation = self.last_conv_activation  # 16x512x4x7

        one_hot = np.zeros(tuple(y_.size()), dtype=np.float32)
        for i in range(len(y)):
            one_hot[i][y[i]] = 1
        one_hot = torch.from_numpy(one_hot).cuda(self.gpu)
        one_hot = torch.sum(one_hot * y_)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.last_conv_gradient[0].cpu().numpy()  # 16x512x4x7
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
            resized_cam = cv2.resize(normalized_cam, (self.model.W, self.model.H), interpolation=cv2.INTER_LINEAR)
            self.checkpoint_image(x[i], y[i], argmax_y_[i], resized_cam, image_checkpoint_dir, name[i])

    @abstractmethod
    def checkpoint_image(self, x, y, y_, resized_cam, image_checkpoint_dir, name):
        pass

class SingleImageGradCAM(GradCAM):
    def __init__(self, model, args):
        self.model = model
        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.last_conv_layer = args.last_conv_layer
        self.last_conv_activation = None
        self.last_conv_gradient = None
        self.register_hook()

    def register_forward_hook(self, module, x, y):
        self.last_conv_activation = y

    def register_backward_hook(self, module, x, y):
        self.last_conv_gradient = y

    def register_hook(self):
        for name, layer in self.model.named_modules():
            if name == self.last_conv_layer:
                layer.register_forward_hook(self.register_forward_hook)
                layer.register_backward_hook(self.register_backward_hook)
                return
        assert False

    def checkpoint_image(self, x, y, y_, resized_cam, image_checkpoint_dir, name):
        if y != 0:
            x = x.cpu().numpy().astype(np.float32) / 255.
            image = np.transpose(x, (1,2,0))
            if image.shape[-1] == 1:
                image = np.tile(image, 3)
            cam = get_color_map(resized_cam, normalize=False)
            comb = cam + image
            comb /= np.max(comb)
            image = (image * 255).astype(np.uint8)
            resized_cam = (resized_cam * 255).astype(np.uint8)
            comb = (comb * 255).astype(np.uint8)
            if name.split('.')[-1] != 'jpg':
                name = '.'.join(name.split('.')[:-1]) + '.jpg'
            skimage.io.imsave('{}/image/{}'.format(image_checkpoint_dir, name), image)
            skimage.io.imsave('{}/cam/{}'.format(image_checkpoint_dir, name), resized_cam)
            skimage.io.imsave('{}/comb/{}'.format(image_checkpoint_dir, name), comb)
