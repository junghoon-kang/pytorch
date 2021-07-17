import os
import json
import glob
import random
import skimage.io
import torch
import numpy as np
import albumentations as A

from vision.annotation import *


__all__ = [
    "ClassificationDataset",
    "SegmentationDataset",
]


class ImageDataset(object):
    """ Meta-class which handles images and their classification/segmentation
    labels.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_one_hot_cla_label(self, label, num_classes):
        """ Converts the classification label into one hot classification label
        format.

        Args:
            label (int): the classification label of an image.

        Returns:
            one_hot_label (numpy.ndarray):
                the one hot label of an image with shape (num_classes,).
        """
        one_hot_label = np.zeros([num_classes], dtype=np.uint8)
        one_hot_label[label] = 1
        return torch.Tensor(one_hot_label).type(torch.uint8)

    def get_one_hot_seg_label(self, label, num_classes):
        """ Converts the segmentation label into one hot segmentation label
        format.

        Args:
            label (numpy.ndarray):
                label image with shape (H,W). The unknown area has the pixel
                value of 0, and the pixel value of each ROI starts from 1 to
                255.

        Returns:
            one_hot_label (numpy.ndarray):
                the one hot segmentation label of an image with shape
                (num_classes,H,W).
        """
        h, w = label.shape
        one_hot_label = np.zeros((num_classes, h, w), dtype=np.uint8)
        for c in range(num_classes):
            one_hot_label[c][np.where(label==c)] = 1
        return torch.Tensor(one_hot_label).type(torch.uint8)


class ClassificationDataset(ImageDataset):
    def __init__(self, annotation, transforms=[], one_hot=False):
        if not isinstance(annotation, (SingleImageAnnotation, MultiImageAnnotation)):
            raise TypeError("type(annotation) must be in {SingleImageAnnotation, MultiImageAnnotation, PairImageAnnotation}.")

        self.annotation = annotation
        self.transforms = A.Compose(transforms)
        self.one_hot = one_hot
        self.num_classes = annotation.num_classes

    def __getitem__(self, index):
        image_filepath, cla_label, seg_label_filepath = self.annotation[index].data

        if isinstance(self.annotation, SingleImageAnnotation):
            image = skimage.io.imread(image_filepath)
        else:
            image = [ skimage.io.imread(path) for path in image_filepath ]

        if seg_label_filepath is None:
            transformed = self.transforms(image=image, cla_label=cla_label)
        else:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
            else:
                seg_label = np.zeros(image.shape[:2], dtype=np.uint8)
            transformed = self.transforms(
                image=image,
                mask=seg_label,
                cla_label=cla_label,
                name=os.path.basename(image_filepath)
            )
        image = transformed["image"]

        if self.one_hot:
            cla_label = self.get_one_hot_cla_label(cla_label, self.num_classes)

        if isinstance(self.annotation, SingleImageAnnotation):
            return image, cla_label, os.path.basename(image_filepath)
        else:
            product_dirpath = os.path.dirname(image_filepath[0])
            return image, cla_label, os.path.basename(product_dirpath)

    def __len__(self):
        return len(self.annotation)


class SegmentationDataset(ImageDataset):
    def __init__(self, annotation, transforms=[], one_hot=False):
        if not isinstance(annotation, (SingleImageAnnotation, MultiImageAnnotation)):
            raise TypeError("type(annotation) must be in {SingleImageAnnotation, MultiImageAnnotation, PairImageAnnotation}.")

        self.annotation = annotation
        self.transforms = A.Compose(transforms)
        self.one_hot = one_hot
        self.num_classes = annotation.num_classes

    def __getitem__(self, index):
        image_filepath, cla_label, seg_label_filepath = self.annotation[index].data

        if isinstance(self.annotation, SingleImageAnnotation):
            image = skimage.io.imread(image_filepath)
        else:
            image = [ skimage.io.imread(path) for path in image_filepath ]

        if seg_label_filepath is None:
            raise ValueError("seg_label_filepath should be str not None.")
        else:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
            else:
                seg_label = np.zeros(image.shape[:2], dtype=np.uint8)
            transformed = self.transforms(
                image=image,
                mask=seg_label,
                cla_label=cla_label,
                name=os.path.basename(image_filepath)
            )
        image, seg_label = transformed["image"], transformed["mask"]

        if self.one_hot:
            seg_label = self.get_one_hot_seg_label(seg_label, self.num_classes)

        if isinstance(self.annotation, SingleImageAnnotation):
            return image, seg_label, os.path.basename(image_filepath)
        else:
            product_dirpath = os.path.dirname(image_filepath[0])
            return image, seg_label, os.path.basename(product_dirpath)

    def __len__(self):
        return len(self.annotation)
