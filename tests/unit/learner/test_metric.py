import os, sys
import pytest
import torch
import torchmetrics
import numpy as np
from skimage.io import imread, imsave

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
import config


def to_one_hot_image(image, num_classes=2):
    h, w = image.shape
    one_hot = np.zeros((num_classes, h, w), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[c][np.where(image==c)] = 1
    return one_hot

def test_IoU():
    path = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
    seg_label_dirpath1 = os.path.join(path, "mask", "labeler.2class")
    seg_label_dirpath2 = os.path.join(path, "mask", "original.2class")

    true = imread(os.path.join(seg_label_dirpath1, "domain1.test.NG.0002.png"))
    pred = imread(os.path.join(seg_label_dirpath2, "domain1.test.NG.0002.png"))

    pred = to_one_hot_image(pred)

    true = torch.from_numpy(true).type(torch.uint8)  # torch.Size([512, 512])
    pred = torch.from_numpy(pred).type(torch.float32)  # torch.Size([2, 512, 512])

    true = torch.unsqueeze(true, axis=0)  # torch.Size([1, 512, 512])
    pred = torch.unsqueeze(pred, axis=0)  # torch.Size([1, 2, 512, 512])

    iou = torchmetrics.IoU(num_classes=2)
    assert np.round( iou(pred, true).item(), 4 ) == 0.7017
