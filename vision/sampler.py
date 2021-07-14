import numpy as np
import torch.utils.data


__all__ = [
    "RandomSampler",
    "WeightedSampler",
    "WeightedSampler2",
]


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
    def __init__(self, annotation):
        self.annotation = annotation

    def __iter__(self):
        indices = list(range(len(self.annotation)))
        return iter(shuffle(indices))

    def __len__(self):
        return len(self.annotation)


class WeightedSampler(torch.utils.data.Sampler):
    def __init__(self, annotation, weights=None):
        self.annotation = annotation
        self.subsets = self.get_subsets(annotation)
        self.weights = self.set_weights(weights, annotation.num_classes)
        self.weighted_subset_sizes = self.get_weighted_subset_sizes(self.subsets, self.weights)

    def __iter__(self):
        indices = []
        for label, weighted_subset_size in enumerate(self.weighted_subset_sizes):
            indices += self.sample_subset(label, weighted_subset_size)
        return iter(shuffle(indices))

    def __len__(self):
        return sum(self.weighted_subset_sizes)

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

    def set_weights(self, weights, num_classes):
        if weights is None:
            return [ 1 for i in range(num_classes) ]
        else:
            assert len(weights) == num_classes
            return weights

    def get_weighted_subset_sizes(self, subsets, weights):
        reduced_subset_sizes = list(
            map(
                lambda tup: int(np.ceil(tup[0]/tup[1])),
                zip(
                    map(lambda subset: len(subset), subsets),  # [397,64]
                    weights  # [3,1]
                )  # [(397,3), (64,1)]
            )  # [133, 64]
        )
        max_unit = max(reduced_subset_sizes)  # 133
        units = [ max_unit if size != 0 else 0 for size in reduced_subset_sizes ]
        return [ w * u for w, u in zip(weights, units) ]

    def sample_subset(self, label, weighted_subset_size):
        subset = self.subsets[label]
        subset_size = len(subset)
        if weighted_subset_size == 0 or subset_size == 0:
            return []
        if weighted_subset_size < subset_size:
            raise ValueError("weighted_subset_size must not be smaller than subset_size")
        q = weighted_subset_size // subset_size
        r = weighted_subset_size % subset_size
        return q * subset + np.random.choice(subset, r, replace=False).tolist()


class WeightedSampler2(torch.utils.data.WeightedRandomSampler):
    def __init__(self, annotation, class_weights=None):
        self.annotation = annotation
        self.class_weights = self.get_class_weights(annotation, class_weights)
        self.class_counts = self.get_class_counts(annotation)
        super(WeightedSampler2, self).__init__(
            self.get_sample_weights(annotation, self.class_weights, self.class_counts),
            len(annotation),
            replacement=True,
            generator=None
        )

    def get_class_weights(self, annotation, class_weights):
        if class_weights is None:
            return [ 1 for i in range(annotation.num_classes) ]
        else:
            assert len(class_weights) == annotation.num_classes
            return class_weights

    def get_class_counts(self, annotation):
        class_counts = []
        for c in range(annotation.num_classes):
            class_counts.append( len([ a for a in annotation if a.cla_label == c ]) )
        return class_counts

    def get_sample_weights(self, annotation, class_weights, class_counts):
        sample_weights = []
        for a in annotation:
            weight = class_weights[a.cla_label]
            count = class_counts[a.cla_label]
            sample_weights.append( weight / count )
        return sample_weights
