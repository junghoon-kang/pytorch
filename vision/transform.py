import torch
import random
import numpy as np
import albumentations as A


__all__ = [
    "RandomCropNearDefect",
    "To3channel",
    "ToTensor",
]


class RandomCropNearDefect(A.DualTransform):
    def __init__(self, size=(128,128), coverage_size=(128,128), fixed=False, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        if not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError("size should be list or tuple.")
        if len(size) != 2:
            raise ValueError("size should be list or tuple of size 2.")
        if not (isinstance(size[0], int) and isinstance(size[1], int)):
            raise TypeError("size should be list or tuple of integers.")

        if not (isinstance(coverage_size, tuple) or isinstance(coverage_size, list)):
            raise TypeError("coverage_size should be list or tuple.")
        if len(coverage_size) != 2:
            raise ValueError("coverage_size should be list or tuple of size 2.")
        if not (isinstance(coverage_size[0], int) and isinstance(coverage_size[1], int)):
            raise TypeError("coverage_size should be list or tuple of integers.")
        if coverage_size[0] <= 0 or coverage_size[1] <= 0:
            raise TypeError("coverage_size should be a positive integer.")
        self.size = size
        self.coverage_size = coverage_size
        self.fixed = fixed
        if self.fixed:
            self.coverage_size = (1,1)

    def get_transform_init_args_names(self):
        return (
            "size",
            "coverage_size",
            "fixed",
        )

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        seg_label = params["mask"]
        if np.sum(seg_label) == 0:
            h, w = seg_label.shape[:2]
            px = random.randint(self.size[0]//2, h - self.size[0]//2)
            py = random.randint(self.size[1]//2, h - self.size[1]//2)
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
        coords = self.get_coords_from_pivot(seg_label, combined_pivot, self.size)
        return {"coords": coords}

    def get_coords_from_pivot(self, image, pivot, size):
        h, w = image.shape[:2]
        if h < size[0] or w < size[1]:
            raise ValueError(f"Requested crop size ({size[0]}, {size[1]}) is larger than the image size ({h}, {w})")
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

    def apply(self, image, **params):
        y_min, x_min, y_max, x_max = params["coords"]
        return A.functional.crop(image, x_min, y_min, x_max, y_max)

    def apply_to_mask(self, image, **params):
        y_min, x_min, y_max, x_max = params["coords"]
        return A.functional.crop(image, x_min, y_min, x_max, y_max)


class To3channel(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1):
        super(To3channel, self).__init__(always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ()

    def apply(self, image, **params):
        if len(image.shape) != 2:
            raise ValueError("image should be 1-channel image")
        return np.dstack((image,)*3)


class ToTensor(A.DualTransform):
    def __init__(self, always_apply=True, p=1):
        super().__init__(always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ()

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def get_params_dependent_on_targets(self, params):
        return {}

    def apply(self, image, **params):
        if image.ndim == 2:
            image = np.expand_dims(image, 2)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image)
        return tensor.contiguous().float()

    def apply_to_mask(self, mask, **params):
        tensor = torch.from_numpy(mask)
        return tensor.contiguous()
