import os
import json
import glob
import random
import skimage.io
import numpy as np
import albumentations as A


__all__ = [
    'SingleImageClassificationDataset',
    'SingleImageSegmentationDataset',
    'MultiImageClassificationDataset',
    'MultiImageSegmentationDataset',
    #'PairImageClassificationDataset',
    #'PairImageSegmentationDataset',
]


class ImageDataset(object):
    """ Meta-class which handles images and their classification/segmentation
    labels.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_subsets(self, samples, num_classes):
        """ Divide samples into the sets of different classes.

        Args:
            samples (list[tuple[str,int,str]]): 
                the list of tuples where each tuple consists of an image path,
                a classification label, and a segmentation label path.
            num_classes (int): the total number of classes.

        Returns:
            subsets (list[list[int]]): 
                the list of class lists where each class list contains the
                index of samples of the same class.
        """
        subsets = []
        for label in range(num_classes):
            subsets.append([])
        for i in range(len(self.samples)):
            image, cla_label, seg_label = self.samples[i]
            subsets[cla_label].append(i)
        return subsets

    def get_one_hot_cla_label(self, label):
        """ Converts the classification label into one hot classification label
        format.

        Args:
            label (int): the classification label of an image.

        Returns:
            one_hot_label (numpy.ndarray):
                the one hot label of an image with shape (num_classes,).
        """
        one_hot_label = np.zeros([self.num_classes], dtype=np.float32)
        one_hot_label[label] = 1
        return one_hot_label

    def get_one_hot_seg_label(self, label):
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
        indices = []
        for i in range(self.num_classes):
            indices.append(np.where(label == i))
        h, w = label.shape if len(label.shape) == 2 else label.shape[1:]
        one_hot_label = np.zeros((self.num_classes, h, w), dtype=np.float32)
        for i in range(self.num_classes):
            one_hot_label[i, indices[i][1], indices[i][2]] = 1
        return one_hot_label


class SingleImageClassificationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath=None, transforms=[], one_hot=False):
        def process_line(line):
            image_filepath = os.path.join(image_dirpath, line)
            cla_label_list = self.annotation['single_image'][line]['class']
            seg_label_filepath  = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
            return [ (image_filepath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, 'r') as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, 'r') as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation['classes'])
        self.transforms = A.Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        image_filepath, cla_label, seg_label_filepath = self.samples[index]
        image = skimage.io.imread(image_filepath)

        if seg_label_filepath is not None:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath)
            else:
                seg_label = np.zeros(image.shape[:2], dtype=np.float32)
            transformed = self.transforms(image=image, mask=seg_label)
        else:
            transformed = self.transforms(image=image)
        image = transformed['image']

        if self.one_hot:
            cla_label = self.get_one_hot_cla_label(cla_label)

        return image, cla_label, os.path.basename(image_filepath)

    def __len__(self):
        return len(self.samples)


class SingleImageSegmentationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath, transforms=[], one_hot=False):
        def process_line(line):
            image_filepath = os.path.join(image_dirpath, line)
            cla_label_list = self.annotation['single_image'][line]['class']
            seg_label_filepath  = os.path.join(seg_label_dirpath, line)
            return [ (image_filepath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, 'r') as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, 'r') as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation['classes'])
        self.transforms = A.Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        image_filepath, cla_label, seg_label_filepath = self.samples[index]
        image = skimage.io.imread(image_filepath)

        if os.path.exists(seg_label_filepath):
            seg_label = skimage.io.imread(seg_label_filepath)
        else:
            seg_label = np.zeros(image.shape[:2], dtype=np.float32)
        transformed = self.transforms(image=image, mask=seg_label)
        image, seg_label = transformed['image'], transformed['mask']

        if self.one_hot:
            seg_label = self.get_one_hot_seg_label(seg_label)

        return image, seg_label, os.path.basename(image_filepath)

    def __len__(self):
        return len(self.samples)


class MultiImageClassificationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath=None, transforms=[], one_hot=False):
        def process_line(line):
            product_dirpath = os.path.join(image_dirpath, line)
            cla_label_list = list( self.annotation['multi_image'][line].values() )[0]['class']  # TODO: suppport independent label for each image
            seg_label_filepath = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
            return [ (product_dirpath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, 'r') as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, 'r') as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation['classes'])
        self.transforms = A.Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        product_dirpath, cla_label, seg_label_filepath = self.samples[index]
        product = [ skimage.io.imread(image_filepath) for image_filepath in sorted(glob.glob(os.path.join(product_dirpath, '*'))) ]

        if seg_label_filepath is not None:
            if os.path.exists(seg_label_filepath):
                seg_label = skimage.io.imread(seg_label_filepath)
            else:
                seg_label = np.zeros(product[0].shape[:2], dtype=np.float32)
            transformed = self.transforms(image=product, mask=seg_label)
        else:
            seg_label = None
            transformed = self.transforms(image=product)
        product = transformed['image']

        if self.one_hot:
            cla_label = self.get_one_hot_cla_label(cla_label)

        return product, cla_label, os.path.basename(product_dirpath)

    def __len__(self):
        return len(self.samples)


class MultiImageSegmentationDataset(ImageDataset):
    def __init__(self, image_dirpath, annotation_filepath, imageset_filepath, seg_label_dirpath, transforms=[], one_hot=False):
        def process_line(line):
            product_dirpath = os.path.join(image_dirpath, line)
            cla_label_list = list( self.annotation['multi_image'][line].values() )[0]['class']  # TODO: suppport independent label for each image
            seg_label_filepath = os.path.join(seg_label_dirpath, line)
            return [ (product_dirpath, cla_label, seg_label_filepath) for cla_label in cla_label_list ]

        with open(annotation_filepath, 'r') as f:
            self.annotation = json.loads(f.read())

        with open(imageset_filepath, 'r') as f:
            self.samples = []
            for line in f.read().splitlines():
                for tup in process_line(line):
                    self.samples.append(tup)

        self.num_classes = len(self.annotation['classes'])
        self.transforms = Compose(transforms)

        self.subsets = self.get_subsets(self.samples, self.num_classes)
        self.one_hot = one_hot

    def __getitem__(self, index):
        product_dirpath, cla_label, seg_label_filepath = self.samples[index]
        product = [ skimage.io.imread(image_filepath) for image_filepath in sorted(glob.glob(os.path.join(product_dirpath, '*'))) ]

        if os.path.exists(seg_label_filepath):
            seg_label = skimage.io.imread(seg_label_filepath)
        else:
            seg_label = np.zeros((image[0].shape[0], image[0].shape[1]), dtype=np.float32)
        transformed = self.transforms(image=product, mask=seg_label)
        product, seg_label = transformed['image'], transformed['mask']

        if self.one_hot:
            cla_label = self.get_one_hot_seg_label(cla_label)

        return product, seg_label, os.path.basename(product_dirpath)

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from albumentations.pytorch import ToTensor

    import os, sys
    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(PATH, '..'))
    import config
    from vision.transform import *

    path = os.path.join(config.DATA_DIR, 'public', 'DAGM', 'original')
    image_dirpath = os.path.join(path, 'image')
    annotation_filepath = os.path.join(path, 'annotation', 'domain1.single_image.2class.json')
    imageset_filepath = os.path.join(path, 'imageset', 'domain1.single_image.2class', 'public', 'ratio', '100%', 'test.txt')
    seg_label_dirpath = os.path.join(path, 'mask', 'original.2class')

    def SingleImageClassificationDataset_test():
        def test1():
            dataset = SingleImageClassificationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                transforms=[]
            )
            x, y, name = dataset[0]
            print(x.shape, y, name)  # (512, 512) 0 domain1.test.OK.0373.png
        test1()

        def test2():
            dataset = SingleImageClassificationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                transforms=[ToTensor()]
            )
            data_loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
            for x, y, z in data_loader:
                print(z)  # ('domain1.test.OK.0373.png', 'domain1.test.NG.0021.png', 'domain1.test.OK.0136.png', 'domain1.test.OK.0227.png')
                print(type(x), x.size())  # <class 'torch.Tensor'> torch.Size([4, 512, 512])
                print(type(y), y.size())  # <class 'torch.Tensor'> torch.Size([4])
                break
        test2()

        def test3():
            dataset = SingleImageClassificationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                transforms=[To3channel(), ToTensor()]
            )
            data_loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
            for x, y, z in data_loader:
                print(z)  # ('domain1.test.OK.0373.png', 'domain1.test.NG.0021.png', 'domain1.test.OK.0136.png', 'domain1.test.OK.0227.png')
                print(type(x), x.size())  # <class 'torch.Tensor'> torch.Size([4, 3, 512, 512])
                print(type(y), y.size())  # <class 'torch.Tensor'> torch.Size([4])
                break
        test3()
    #SingleImageClassificationDataset_test()

    def SingleImageSegmentationDataset_test():
        def test1():
            dataset = SingleImageSegmentationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                seg_label_dirpath,
                transforms=[]
            )
            x, y, z = dataset[0]
            print(x.shape, y.shape, z)  # (512, 512) (512, 512) domain1.test.OK.0373.png
        test1()

        def test2():
            dataset = SingleImageSegmentationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                seg_label_dirpath,
                transforms=[ToTensor()]
            )
            x, y, z = dataset[0]
            print(x.shape, y.shape, z)  # torch.Size([512, 512]) torch.Size([1, 512, 512]) domain1.test.OK.0373.png
        test2()

        def test3():
            dataset = SingleImageSegmentationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                seg_label_dirpath,
                transforms=[ToTensor()],
                one_hot=True
            )
            x, y, z = dataset[0]
            print(x.shape, y.shape, z)  # torch.Size([512, 512]) (2, 512, 512) domain1.test.OK.0373.png
        test3()

        def test4():
            dataset = SingleImageSegmentationDataset(
                image_dirpath,
                annotation_filepath,
                imageset_filepath,
                seg_label_dirpath,
                transforms=[ToTensor()]
            )
            data_loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )
            for x, y, name in data_loader:
                print(name)  # ('domain1.test.OK.0490.png', 'domain1.test.OK.0349.png', 'domain1.test.OK.0196.png', 'domain1.test.OK.0245.png')
                print(type(x), x.size())  # <class 'torch.Tensor'> torch.Size([4, 512, 512])
                print(type(y), y.size())  # <class 'torch.Tensor'> torch.Size([4, 1, 512, 512])
                break
        test4()
    #SingleImageSegmentationDataset_test()
