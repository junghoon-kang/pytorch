import os
import json
import glob
import random
import skimage.io
import torch
import numpy as np
import albumentations as A


__all__ = [
    "ClassificationDataset",
    "SingleImageClassificationDataset",
    "SingleImageSegmentationDataset",
    "MultiImageClassificationDataset",
    "MultiImageSegmentationDataset",
    #"PairImageClassificationDataset",
    #"PairImageSegmentationDataset",
]


class ImageDataset(object):
    """ Meta-class which handles images and their classification/segmentation
    labels.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_subsets(self, annotation):
        """ Divide annotation into the sets of different classes.

        Args:
            annotation (list[tuple[str,int,str]]): 
                the list of tuples where each tuple consists of an image path,
                a classification label, and a segmentation label path.
            num_classes (int): the total number of classes.

        Returns:
            subsets (list[list[int]]): 
                the list of class lists where each class list contains the
                index of annotation of the same class.
        """
        subsets = []
        for label in range(annotation.num_classes):
            subsets.append([])
        for i in range(len(annotation)):
            subsets[annotation[i].cla_label].append(i)
        return subsets

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
        self.annotation = annotation
        self.transforms = A.Compose(transforms)
        self.subsets = self.get_subsets(annotation)
        self.one_hot = one_hot

    def __getitem__(self, index):
        image_filepath = self.annotation[index].image
        cla_label = self.annotation[index].cla_label
        seg_label_filepath = self.annotation[index].seg_label

        if isinstance(self.annotation, SingleImageAnnotation):
            image = skimage.io.imread(image_filepath)
        elif isinstance(self.annotation, MultiImageAnnotation):
            image = [ skimage.io.imread(p) for p in image_filepath ]
        else:
            raise ValueError

        if seg_label_filepath is not None:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
            else:
                seg_label = np.zeros(image.shape[:2], dtype=np.uint8)
            transformed = self.transforms(image=image, mask=seg_label)
        else:
            transformed = self.transforms(image=image)
        image = transformed["image"]

        if self.one_hot:
            cla_label = self.get_one_hot_cla_label(cla_label, self.annotation.num_classes)

        if isinstance(self.annotation, SingleImageAnnotation):
            return image, cla_label, os.path.basename(image_filepath)
        elif isinstance(self.annotation, MultiImageAnnotation):
            return image, cla_label, os.path.basename(os.path.dirname(image_filepath))
        else:
            raise ValueError

    def __len__(self):
        return len(self.annotation)

class SingleImageClassificationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath=None, transforms=[], one_hot=False):
        def process_line(line):
            image_filepath = os.path.join(image_dirpath, line)
            cla_label_list = self.annotation["single_image"][line]["class"]
            seg_label_filepath  = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
            return [ (image_filepath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, "r") as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, "r") as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation["classes"])
        self.transforms = A.Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        image_filepath, cla_label, seg_label_filepath = self.samples[index]
        image = skimage.io.imread(image_filepath)

        if seg_label_filepath is not None:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
            else:
                seg_label = np.zeros(image.shape[:2], dtype=np.uint8)
            transformed = self.transforms(image=image, mask=seg_label)
        else:
            transformed = self.transforms(image=image)
        image = transformed["image"]

        if self.one_hot:
            cla_label = self.get_one_hot_cla_label(cla_label, self.num_classes)

        return image, cla_label, os.path.basename(image_filepath)

    def __len__(self):
        return len(self.samples)


class MultiImageClassificationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath=None, transforms=[], one_hot=False):
        def process_line(line):
            product_dirpath = os.path.join(image_dirpath, line)
            cla_label_list = list( self.annotation["multi_image"][line].values() )[0]["class"]  # TODO: suppport independent label for each image
            seg_label_filepath = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
            return [ (product_dirpath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, "r") as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, "r") as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation["classes"])
        self.transforms = A.Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        product_dirpath, cla_label, seg_label_filepath = self.samples[index]
        product = [ skimage.io.imread(image_filepath) for image_filepath in sorted(glob.glob(os.path.join(product_dirpath, "*"))) ]

        if seg_label_filepath is not None:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
            else:
                seg_label = np.zeros(product[0].shape[:2], dtype=np.uint8)
            transformed = self.transforms(image=product, mask=seg_label)
        else:
            seg_label = None
            transformed = self.transforms(image=product)
        product = transformed["image"]

        if self.one_hot:
            cla_label = self.get_one_hot_cla_label(cla_label, self.num_classes)

        return product, cla_label, os.path.basename(product_dirpath)

    def __len__(self):
        return len(self.samples)


class SingleImageSegmentationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath, transforms=[], one_hot=False):
        def process_line(line):
            image_filepath = os.path.join(image_dirpath, line)
            cla_label_list = self.annotation["single_image"][line]["class"]
            seg_label_filepath  = os.path.join(seg_label_dirpath, line)
            return [ (image_filepath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, "r") as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, "r") as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation["classes"])
        self.transforms = A.Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        image_filepath, cla_label, seg_label_filepath = self.samples[index]
        image = skimage.io.imread(image_filepath)

        if os.path.exists(seg_label_filepath):
            seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
        else:
            seg_label = np.zeros(image.shape[:2], dtype=np.uint8)
        transformed = self.transforms(image=image, mask=seg_label)
        image, seg_label = transformed["image"], transformed["mask"]

        if self.one_hot:
            seg_label = self.get_one_hot_seg_label(seg_label, self.num_classes)

        return image, seg_label, os.path.basename(image_filepath)

    def __len__(self):
        return len(self.samples)


class MultiImageSegmentationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath, transforms=[], one_hot=False):
        def process_line(line):
            product_dirpath = os.path.join(image_dirpath, line)
            cla_label_list = list( self.annotation["multi_image"][line].values() )[0]["class"]  # TODO: suppport independent label for each image
            seg_label_filepath = os.path.join(seg_label_dirpath, line)
            return [ (product_dirpath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, "r") as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, "r") as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation["classes"])
        self.transforms = Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        product_dirpath, cla_label, seg_label_filepath = self.samples[index]
        product = [ skimage.io.imread(image_filepath) for image_filepath in sorted(glob.glob(os.path.join(product_dirpath, "*"))) ]

        if os.path.exists(seg_label_filepath):
            seg_label = skimage.io.imread(seg_label_filepath).astype(np.uint8)
        else:
            seg_label = np.zeros((image[0].shape[0], image[0].shape[1]), dtype=np.uint8)
        transformed = self.transforms(image=product, mask=seg_label)
        product, seg_label = transformed["image"], transformed["mask"]

        if self.one_hot:
            seg_label = self.get_one_hot_seg_label(cla_label, self.num_classes)

        return product, seg_label, os.path.basename(product_dirpath)

    def __len__(self):
        return len(self.samples)
