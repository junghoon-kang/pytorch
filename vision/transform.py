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
    "ZoomIn",
    "RandomCrop",
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


class ZoomIn(A.DualTransform):
    def __init__(
        self,
        scale_limit=(1.,1.2),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=0.5
    ):
        super(ZoomIn, self).__init__(always_apply=always_apply, p=p)
        self.scale_limit = A.to_tuple(scale_limit, bias=0)
        self.interpolation = interpolation

    def get_params(self):
        return { "scale": np.random.uniform(self.scale_limit[0], self.scale_limit[1]) }

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
        h, w = image.shape[:2]
        image = A.scale(image, params["scale"], self.interpolation)
        image = A.center_crop(image, h, w)
        return image

    def get_transform_init_args_names(self):
        return {
            "scale_limit": self.scale_limit,
            "interpolation": self.interpolation,
        }


class RandomCrop(A.DualTransform):
    def __init__(
        self,
        size=(128, 128),
        coverage_size=(64, 64),
        fixed=False,
        always_apply=True,
        p=1.0
    ):
        __is_sequence_of_two_positive_integers(size, "size")
        __is_sequence_of_two_positive_integers(coverage_size, "coverage_size")
        if coverage_size[0] > size[0] or coverage_size[1] > size[1]:
            raise ValueError("coverage_size must be smaller than or equal to size")

        super(RandomCrop, self).__init__(always_apply=always_apply, p=p)
        self.size = size
        self.coverage_size = coverage_size
        self.fixed = fixed
        if self.fixed:
            self.coverage_size = (1,1)

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        seg_label = params["mask"]
        h, w = seg_label.shape[:2]
        if h < self.size[0] or w < self.size[1]:
            raise ValueError(f"Requested crop size ({size[0]}, {size[1]}) is larger than the image size ({h}, {w})")

        if np.sum(seg_label) == 0:
            px = random.randint(self.size[0]//2, h - self.size[0]//2)
            py = random.randint(self.size[1]//2, w - self.size[1]//2)
            combined_pivot = (px, py)  # middle pivot
        else:
            # pick random point from the coverage box
            coverage_pivot = tuple(map(lambda l: random.randint(0, l-1), self.coverage_size))
            dh = coverage_pivot[0] - (self.coverage_size[0] - 1) // 2
            dw = coverage_pivot[1] - (self.coverage_size[1] - 1) // 2
            # pick random point from defect pixels
            indices = np.where(seg_label != 0)  # TODO: handle the case when there are more than one defect classes by random cropping near the specific defect class
            if self.fixed:
                i = 0
            else:
                i = random.randint(0, len(indices[0])-1)
            defect_pivot = (indices[0][i], indices[1][i])
            combined_pivot = (defect_pivot[0] - dh, defect_pivot[1] - dw)
        coords = self.__get_coords_from_pivot(combined_pivot, self.size)
        return {"coords": coords}

    def __get_coords_from_pivot(self, pivot, size):
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
        ],
        always_apply=True,
        p=1.0
    ):
        self.__sanity_check_patterns(patterns)
        super(Cutout, self).__init__(always_apply=always_apply, p=p)
        n = len(patterns) + 1
        self.patterns = [ ((i+1)/n, patterns[i]) for i in range(n-1) ] + [(1,None)]

    def __santiy_check_patterns(self, patterns):
        for pattern in patterns:
            if "name" not in pattern:
                raise ValueError("pattern must contain name as a key")
            name = pattern["name"]
            if name == "rectangle":
                if "size" not in pattern:
                    raise ValueError("retangle pattern must contain size as a key.")
                __is_sequence_of_two_positive_integers(size, "size")
            elif name == "roi":
                pass
            else:
                raise ValueError("pattern name must be one of the following: ['rectangle', 'roi']")

    def get_params(self):
        p = np.random.uniform()
        for bound, pattern in self.patterns:
            if p <= bound:
                return { "pattern": pattern }

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        if params["operation"] is None:
            return { "indices": None }

        op = params["operation"]
        if op["name"] == "rectangle":
            pass

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def apply(self, image, **params):
        if params["pattern"] is None:
            return image

    def apply_to_mask(self, mask, **params):
        if params["pattern"] is None:
            return mask

    def get_transform_init_args_names(self):
        return { "operations": self.operations }

def cutout(image, cla_label, seg_label, operation):
    if name == "rectangle":
        size = __get_size_for_rectangular_cutout(operation)
        max_coverage_ratio = __get_max_coverage_ratio_for_cutout(operation)
        indices = __get_seg_label_indices_for_rectangular_cutout(seg_label, cla_label, size, max_coverage_ratio)
        image[indices] = 0
        seg_label[indices] = 0
    elif name == "roi":
        if cla_label != 0:
            image[np.where(seg_label == 0)] = 0
    else:
        raise ValueError("name must be one of the following: ['rectangle', 'circle', 'roi']")
    return np.ascontiguousarray(image), cla_label, np.ascontiguousarray(seg_label)


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


def __is_sequence_of_two_positive_integers(seq, name=""):
    if not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be list or tuple")
    if len(seq) != 2:
        raise TypeError(f"{name} must be list or tuple of size 2")
    if not (isinstance(seq[0], int) and isinstance(seq[1], int)):
        raise TypeError(f"{name} must be list or tuple of integers")
    if seq[0] <= 0 or seq[1] <= 0:
        raise ValueError(f"{name} must be a positive integer")


if __name__ == "__main__":
    from IPython import embed; embed(); assert False
