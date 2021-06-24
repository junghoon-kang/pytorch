import cv2
import torch
import random
import numpy as np
import albumentations as A
from typing import Sequence

import numpy as np
from skimage.io import imsave


__all__ = [
    "Rotate90",
    "RandomCrop",
    "Cutout",
    "To3channel",
    "ToTensor",
]


class Rotate90(A.DualTransform):
    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def apply(self, image, **params):
        return np.ascontiguousarray(np.rot90(image, 1))

    def get_transform_init_args_names(self):
        return {}


class RandomCrop(A.DualTransform):
    def __init__(
        self,
        size=(128, 128),
        coverage_size=(64, 64),
        fixed=False,
        always_apply=True,
        p=1.0
    ):
        is_sequence_of_two_positive_numbers(size, "size")
        is_sequence_of_two_positive_numbers(coverage_size, "coverage_size")
        if coverage_size[0] > size[0] or coverage_size[1] > size[1]:
            raise ValueError("coverage_size must be smaller than or equal to size")

        super(RandomCrop, self).__init__(always_apply=always_apply, p=p)
        self.size = tuple(int(size[0]), int(size[1]))
        self.coverage_size = tuple(int(coverage_size[0]), int(coverage_size[1]))
        self.fixed = fixed
        if self.fixed:
            self.coverage_size = (1,1)

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["mask", "cla_label"]

    def get_params_dependent_on_targets(self, params):
        cla_label = params["cla_label"]
        seg_label = params["mask"]
        h, w = seg_label.shape[:2]
        if h < self.size[0] or w < self.size[1]:
            raise ValueError(f"Requested crop size ({size[0]}, {size[1]}) is larger than the image size ({h}, {w})")

        if cla_label == 0:
            px = random.randint(self.size[0]//2, h - self.size[0]//2)
            py = random.randint(self.size[1]//2, w - self.size[1]//2)
            combined_pivot = (px, py)  # middle pivot
        else:
            # pick random point from the coverage box
            coverage_pivot = tuple(map(lambda l: random.randint(0, l-1), self.coverage_size))
            dh = coverage_pivot[0] - (self.coverage_size[0] - 1) // 2
            dw = coverage_pivot[1] - (self.coverage_size[1] - 1) // 2
            # pick random point from defect pixels
            indices = np.where(seg_label == cla_label)  # TODO: handle the case when there are more than one defect classes by random cropping near the specific defect class
            if self.fixed:
                i = 0
            else:
                i = random.randint(0, len(indices[0])-1)
            defect_pivot = (indices[0][i], indices[1][i])
            combined_pivot = (defect_pivot[0] - dh, defect_pivot[1] - dw)
        coords = self.__get_coords_from_pivot(seg_label, combined_pivot, self.size)
        return {"coords": coords}

    def __get_coords_from_pivot(self, seg_label, pivot, size):
        h, w = seg_label.shape[:2]
        h1 = pivot[0] - (size[0] - 1) // 2  # set (h1, w1) to be the top-left point of the patch
        w1 = pivot[1] - (size[1] - 1) // 2
        h2 = h1 + size[0] - 1  # set (h2, w2) to be the bottom-right point of the patch
        w2 = w1 + size[1] - 1
        if h1 < 0:
            delta = -h1
            h1 = 0
            h2 += delta
        if w1 < 0:
            delta = -w1
            w1 = 0
            w2 += delta
        if h2 > h - 1:
            delta = h2 - (h - 1)
            h1 -= delta
            h2 = h - 1
        if w2 > w - 1:
            delta = w2 - (w - 1)
            w1 -= delta
            w2 = w - 1
        return (h1, w1, h2+1, w2+1)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def apply(self, image, **params):
        y_min, x_min, y_max, x_max = params["coords"]
        return A.functional.crop(image, x_min, y_min, x_max, y_max)

    def get_transform_init_args_names(self):
        return {
            "size": self.size,
            "coverage_size": self.coverage_size,
            "fixed": self.fixed,
        }


class Cutout(A.DualTransform):
    def __init__(
        self,
        patterns=[
            dict(name="rectangle", size=(64,64), max_coverage_ratio=0.5),
            dict(name="roi"),
        ],
        always_apply=False,
        p=0.5
    ):
        self.__sanity_check_patterns(patterns)
        super(Cutout, self).__init__(always_apply=always_apply, p=p)
        self.patterns = [ ((i+1)/len(patterns), pattern) for i, pattern in enumerate(patterns) ]

    def __sanity_check_patterns(self, patterns):
        if len(patterns) == 0:
            raise ValueError("the length of patterns must be greater than 0")
        for pattern in patterns:
            if "name" not in pattern:
                raise ValueError("pattern must contain name as a key")
            name = pattern["name"]
            if name == "rectangle":
                if "size" not in pattern:
                    raise ValueError("retangle pattern must contain size as a key.")
                is_sequence_of_two_positive_numbers(pattern["size"], "size")
                if "max_coverage_ratio" not in pattern:
                    raise ValueError("pattern must contain max_coverage_ratio as a key.")
                if not (0 <= pattern["max_coverage_ratio"] <= 1):
                    raise ValueError("max_coverage_ratio must be between 0 and 1.")
            elif name == "roi":
                pass
            else:
                raise ValueError("pattern name must be one of the following: ['rectangle', 'roi']")

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["mask", "cla_label"]

    def get_params_dependent_on_targets(self, params):
        p = np.random.uniform()
        for bound, pattern in self.patterns:
            if p <= bound:
                break

        cla_label = params["cla_label"]
        seg_label = params["mask"]
        if pattern["name"] == "rectangle":
            indices = self.__get_seg_label_indices_for_rectangular_cutout(seg_label, cla_label, pattern["size"], pattern["max_coverage_ratio"])
        elif pattern["name"] == "roi":
            if cla_label != 0:
                indices = np.where(seg_label == 0)
            else:
                indices = None
        return { "indices": indices }

    def __get_seg_label_indices_for_rectangular_cutout(self, seg_label, cla_label, size, max_coverage_ratio):
        indices = self.__get_seg_label_indices_for_rectangular_cutout_helper(seg_label, size)
        while not self.__is_valid_seg_label_indices_for_rectangular_cutout(seg_label, cla_label, indices, max_coverage_ratio):
            indices = self.__get_seg_label_indices_for_rectangular_cutout_helper(seg_label, size)
        return indices

    def __is_valid_seg_label_indices_for_rectangular_cutout(self, seg_label, cla_label, indices, max_coverage_ratio):
        num_total_pixels = len(np.where(seg_label == cla_label)[0])
        num_pixels = len(np.where(seg_label[indices] == cla_label)[0])
        return num_pixels / num_total_pixels <= max_coverage_ratio

    def __get_seg_label_indices_for_rectangular_cutout_helper(self, seg_label, size):
        h, w = seg_label.shape
        center = (np.random.randint(0, h), np.random.randint(0, w))
        indices = []
        t = center[0] - int(np.floor(size[0]/2))
        b = center[0] + int(np.ceil(size[0]/2))
        l = center[1] - int(np.floor(size[1]/2))
        r = center[1] + int(np.ceil(size[1]/2))
        if t < 0: t = 0
        if b > h: b = h
        if l < 0: l = 0
        if r > w: r = w
        indices = np.meshgrid(np.arange(t, b), np.arange(l, r), indexing='ij')
        return tuple(indices)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def apply(self, image, **params):
        result = np.copy(image)
        if params["indices"] is None:
            return result

        result[params["indices"]] = 0
        return result

    def get_transform_init_args_names(self):
        return { "patterns": self.patterns }


class To3channel(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1):
        super(To3channel, self).__init__(always_apply=always_apply, p=p)

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets(self):
        return { "image": self.apply }

    def apply(self, image, **params):
        if image.ndim == 2:
            return np.dstack((image,)*3)
        elif image.ndim == 3:
            c = image.shape[2]
            if c == 1:
                return np.dstack((image,)*3)
            elif c == 3:
                return image
            elif c == 4:
                return image[:,:,:3]
            else:
                raise ValueError(f"Number of channels of input image must be 1, 3, or 4. num_channels = {c}.")
        else:
            raise ValueError(f"Dimension of input image must be 2 or 3. image.ndim = {image.ndim}.")

    def get_transform_init_args_names(self):
        return {}


class ToTensor(A.DualTransform):
    def __init__(self, always_apply=True, p=1):
        super(ToTensor, self).__init__(always_apply=always_apply, p=p)

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def apply(self, image, **params):
        if image.ndim == 2:
            image = np.expand_dims(image, 2)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image)
        return tensor.contiguous().float()

    def apply_to_mask(self, mask, **params):
        tensor = torch.from_numpy(mask)
        return tensor.contiguous()

    def get_transform_init_args_names(self):
        return {}


def is_sequence_of_two_positive_numbers(seq, name=""):
    if not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be list or tuple")
    if len(seq) != 2:
        raise TypeError(f"{name} must be list or tuple of size 2")
    if not (isinstance(seq[0], (int, float)) and isinstance(seq[1], (int, float))):
        raise TypeError(f"{name} must be list or tuple of integers or floats")
    if seq[0] <= 0 or seq[1] <= 0:
        raise ValueError(f"{name} must be greater than 0")


if __name__ == "__main__":
    from IPython import embed; embed(); assert False
