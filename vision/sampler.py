import math
import random
import numpy as np
import torch.utils.data
import torch.distributed


__all__ = ["RandomSampler", "WeightedSampler"]

def shuffle(*arys):
    """ Shuffles multiple input arrays in the same order.

    Args:
        *arys (list[param]): list of arrays
    """
    assert np.mean( list(map( lambda ary: len(ary), arys )) ) == len(arys[0])  # all arrays should have the same length
    perm = np.random.permutation(len(arys[0]))
    shuffled = list(map( lambda ary: np.array(ary)[perm], arys ))
    if len(shuffled) == 1:
        return shuffled[0]
    return shuffled

class RandomSampler(torch.utils.data.Sampler):
    def __init__(self):
        pass

    def __call__(self, dataset):
        self.dataset = dataset
        return self

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        return iter(shuffle(indices))

    def __len__(self):
        return len(self.dataset)

class WeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights=None):
        self.weights = weights

    def __call__(self, dataset):
        self.dataset = dataset
        self.subsets = dataset.subsets
        self.weights = self.set_weights(self.weights, self.dataset.num_classes)
        self.weighted_subset_sizes = self.get_weighted_subset_sizes()
        return self

    def __iter__(self):
        indices = []
        for label, weighted_subset_size in enumerate(self.weighted_subset_sizes):
            indices += self.sample_subset(label, weighted_subset_size)
        return iter(shuffle(indices))

    def __len__(self):
        return sum(self.weighted_subset_sizes)

    def set_weights(self, weights, num_classes):
        if weights is None:
            return [ 1 for i in range(num_classes) ]
        else:
            assert len(weights) == num_classes
            return weights

    def get_weighted_subset_sizes(self):
        reduced_subset_sizes = list(
            map(
                lambda tup: int(np.ceil(tup[0]/tup[1])),
                zip(
                    map(lambda subset: len(subset), self.subsets),  # [397,64]
                    self.weights  # [3,1]
                )  # [(397,3), (64,1)]
            )  # [133, 64]
        )
        max_unit = max(reduced_subset_sizes)  # 133
        units = [ max_unit if size != 0 else 0 for size in reduced_subset_sizes ]
        return [ w * u for w, u in zip(self.weights, units) ]

    def sample_subset(self, label, weighted_subset_size):
        subset = self.subsets[label]
        subset_size = len(subset)
        if weighted_subset_size == 0 or subset_size == 0:
            return []
        if weighted_subset_size < subset_size:
            raise ValueError("weighted_subset_size should not be smaller than subset_size")
        q = weighted_subset_size // subset_size
        r = weighted_subset_size % subset_size
        return q * subset + random.sample(subset, r)


if __name__ == "__main__":
    import os, sys
    from albumentations.pytorch import ToTensor

    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(PATH, ".."))
    import config
    from vision.dataset import *
    from vision.transform import *

    def WeightedSampler_test():
        data_dir = os.path.join(config.DATA_DIR, "public", "DAGM", "original")
        image_path = os.path.join(data_dir, "image")
        annotation = os.path.join(data_dir, "annotation", "domain1.single_image.2class.json")
        imageset   = os.path.join(data_dir, "imageset", "domain1.single_image.2class")
        data_file  = os.path.join(imageset, "public", "ratio", "100%", "train.1.txt")
        dataset  = SingleImageClassificationDataset(image_path, annotation, data_file, transforms=[ToTensor()])

        sampler = WeightedSampler(weights=[3,1])(dataset)
        print("subset_sizes")
        for label, subset in enumerate(sampler.subsets):
            print("{}: {}".format(label, len(subset)))  # 0: 397, 1: 64
        print("weighted_subset_sizes")
        for label, weighted_subset_size in enumerate(sampler.weighted_subset_sizes):
            print("{}: {}".format(label, weighted_subset_size))  # 0: 399, 1: 133
    #WeightedSampler_test()
