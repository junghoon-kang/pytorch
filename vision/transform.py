import math
import cv2
import PIL
import torch
import numpy as np
import albumentations as A
from typing import Sequence

import numpy as np
from skimage.io import imsave


__all__ = [
    "ToTensor",
    "To3channel",
    "Rotate90",
    "RandomCrop",
    "Cutout",
    "Distort",
]


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
        self.size = ( int(size[0]), int(size[1]) )
        self.coverage_size = ( int(coverage_size[0]), int(coverage_size[1]) )
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
            px = np.random.randint(self.size[0]//2, h - self.size[0]//2 + 1)
            py = np.random.randint(self.size[1]//2, w - self.size[1]//2 + 1)
            combined_pivot = (px, py)  # middle pivot
        else:
            # pick random point from the coverage box
            coverage_pivot = tuple(map(lambda l: np.random.randint(0, l), self.coverage_size))
            dh = coverage_pivot[0] - (self.coverage_size[0] - 1) // 2
            dw = coverage_pivot[1] - (self.coverage_size[1] - 1) // 2
            # pick random point from defect pixels
            indices = np.where(seg_label == cla_label)  # TODO: handle the case when there are more than one defect classes by random cropping near the specific defect class
            if self.fixed:
                i = 0
            else:
                i = np.random.randint(0, len(indices[0]))
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


class Distort(A.DualTransform):
    def __init__(
        self,
        grid_size=(8,8),
        magnitude=8,
        always_apply=False,
        p=0.5
    ):
        super(Distort, self).__init__(always_apply=always_apply, p=p)
        self.grid_size = grid_size
        self.magnitude = magnitude

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        boxes = self.__get_boxes(params["image"], self.grid_size)
        quads = self.__boxes_to_quads(boxes)
        (
            inner_tiles,
            boundary_tiles_first_row,
            boundary_tiles_last_row,
            boundary_tiles_first_col,
            boundary_tiles_last_col,
        ) = self.__get_adjacent_tiles_indices(self.grid_size)

        for a, b, c, d in inner_tiles:
            dx = np.random.randint(-self.magnitude, self.magnitude + 1)
            dy = np.random.randint(-self.magnitude, self.magnitude + 1)

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[a]
            quads[a] = [
                x1, y1,
                x2, y2,
                x3 + dx, y3 + dy,
                x4, y4
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[b]
            quads[b] = [
                x1, y1,
                x2 + dx, y2 + dy,
                x3, y3,
                x4, y4
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[c]
            quads[c] = [
                x1, y1,
                x2, y2,
                x3, y3,
                x4 + dx, y4 + dy
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[d]
            quads[d] = [
                x1 + dx, y1 + dy,
                x2, y2,
                x3, y3,
                x4, y4
            ]

        for a, b in boundary_tiles_first_row:
            dx = np.random.randint(-self.magnitude, self.magnitude + 1)

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[a]
            quads[a] = [
                x1, y1,
                x2, y2,
                x3, y3,
                x4 + dx, y4
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[b]
            quads[b] = [
                x1 + dx, y1,
                x2, y2,
                x3, y3,
                x4, y4
            ]

        for a, b in boundary_tiles_last_row:
            dx = np.random.randint(-self.magnitude, self.magnitude + 1)

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[a]
            quads[a] = [
                x1, y1,
                x2, y2,
                x3 + dx, y3,
                x4, y4
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[b]
            quads[b] = [
                x1, y1,
                x2 + dx, y2,
                x3, y3,
                x4, y4
            ]

        for a, b in boundary_tiles_first_col:
            dy = np.random.randint(-self.magnitude, self.magnitude + 1)

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[a]
            quads[a] = [
                x1, y1,
                x2, y2 + dy,
                x3, y3,
                x4, y4
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[b]
            quads[b] = [
                x1, y1 + dy,
                x2, y2,
                x3, y3,
                x4, y4
            ]

        for a, b in boundary_tiles_last_col:
            dy = np.random.randint(-self.magnitude, self.magnitude + 1)

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[a]
            quads[a] = [
                x1, y1,
                x2, y2,
                x3, y3 + dy,
                x4, y4
            ]

            x1, y1, x2, y2, x3, y3, x4, y4 = quads[b]
            quads[b] = [
                x1, y1,
                x2, y2,
                x3, y3,
                x4, y4 + dy
            ]

        return {
            "boxes": boxes,
            "quads": quads
        }

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask
        }

    def apply(self, image, **params):
        image = PIL.Image.fromarray(image)
        image = image.transform(
            image.size,
            PIL.Image.MESH,
            zip(params["boxes"], params["quads"]),
            resample=PIL.Image.BICUBIC
        )
        return np.ascontiguousarray(np.array(image))

    def apply_to_mask(self, mask, **params):
        mask = PIL.Image.fromarray(mask)
        mask = mask.transform(
            mask.size,
            PIL.Image.MESH,
            zip(params["boxes"], params["quads"]),
            resample=PIL.Image.NEAREST
        )
        return np.ascontiguousarray(np.array(mask))

    def __get_boxes(self, image, grid_size):
        H, W = image.shape[:2]
        h, w = grid_size

        tile_width = int(math.floor(W / float(w)))
        tile_height = int(math.floor(H / float(h)))
        last_tile_width = W - (tile_width * (w - 1))
        last_tile_height = H - (tile_height * (h - 1))

        # Get a box for each tile, where the box is 
        # (x_min, y_min, x_max, y_max) of the tile.
        boxes = []
        for i in range(h):
            for j in range(w):
                if i == (h - 1) and j == (w - 1):
                    boxes.append([
                        j * tile_width,
                        i * tile_height,
                        j * tile_width + last_tile_width,
                        i * tile_height + last_tile_height
                    ])
                elif i == (h - 1):
                    boxes.append([
                        j * tile_width,
                        i * tile_height,
                        j * tile_width + tile_width,
                        i * tile_height + last_tile_height
                    ])
                elif j == (w - 1):
                    boxes.append([
                        j * tile_width,
                        i * tile_height,
                        j * tile_width + last_tile_width,
                        i * tile_height + tile_height
                    ])
                else:
                    boxes.append([
                        j * tile_width,
                        i * tile_height,
                        j * tile_width + tile_width,
                        i * tile_height + tile_height
                    ])
        return boxes

    def __boxes_to_quads(self, boxes):
        # For quadrilateral warp, need to specify the four corners:
        # NW, SW, SE, and NE.
        quads = []
        for x1, y1, x2, y2 in boxes:
            quads.append([x1, y1, x1, y2, x2, y2, x2, y1])
        return quads

    def __get_adjacent_tiles_indices(self, grid_size):
        h, w = grid_size

        last_col_indices = []
        for i in range(h):
            last_col_indices.append(w*i + (w-1))

        last_row_indices = list(range((w*h)-w, w*h))

        inner_tiles = []
        for i in range((h * w) - 1):
            if i not in last_row_indices and i not in last_col_indices:
                inner_tiles.append([i, i+1, i+w, i+1+w])

        first_col_indices = []
        for i in range(h):
            first_col_indices.append(w*i)

        first_row_indices = list(range(w))

        boundary_tiles_first_row = []
        for i in range(len(first_row_indices)-1):
            boundary_tiles_first_row.append( [first_row_indices[i], first_row_indices[i+1]] )

        boundary_tiles_last_row = []
        for i in range(len(last_row_indices)-1):
            boundary_tiles_last_row.append( [last_row_indices[i], last_row_indices[i+1]] )

        boundary_tiles_first_col = []
        for i in range(len(first_col_indices)-1):
            boundary_tiles_first_col.append( [first_col_indices[i], first_col_indices[i+1]] )

        boundary_tiles_last_col = []
        for i in range(len(last_col_indices)-1):
            boundary_tiles_last_col.append( [last_col_indices[i], last_col_indices[i+1]] )

        return (
            inner_tiles,
            boundary_tiles_first_row,
            boundary_tiles_last_row,
            boundary_tiles_first_col,
            boundary_tiles_last_col,
        )

    def get_transform_init_args_names(self):
        return {
            "grid_size": self.grid_size,
            "magnitude": self.magnitude,
        }


def is_sequence_of_two_positive_numbers(seq, name=""):
    if not isinstance(seq, Sequence):
        raise TypeError(f"{name} must be list or tuple")
    if len(seq) != 2:
        raise TypeError(f"{name} must be list or tuple of size 2")
    if not (isinstance(seq[0], (int, float)) and isinstance(seq[1], (int, float))):
        raise TypeError(f"{name} must be list or tuple of integers or floats")
    if seq[0] <= 0 or seq[1] <= 0:
        raise ValueError(f"{name} must be greater than 0")
