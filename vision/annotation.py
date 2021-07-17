import os
import glob
import json
from typing import Optional, List, Union


__all__ = [
    "SingleImageAnnotation",
    "MultiImageAnnotation",
    #PairImageAnnotation",
]


IMAGE_EXTS = [
    "bmp",
    "png",
    "jpg",
    "jpeg",
]


class ImageAnnotationSingleton(object):
    def __init__(
        self,
        image: Union[str, List[str]],
        cla_label: int=None,
        seg_label: Optional[str]=None
    ):
        self._image = image
        self._cla_label = cla_label
        self._seg_label = seg_label

    def __repr__(self):
        return f"(image='{self.image}', cla_label='{self.cla_label}', seg_label='{self.seg_label}')"

    @property
    def image(self):
        return self._image

    @property
    def cla_label(self):
        return self._cla_label

    @property
    def seg_label(self):
        return self._seg_label

    @property
    def data(self):
        return self._image, self._cla_label, self._seg_label

class SingleImageAnnotation(list):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def from_research_format(
        self,
        image_dirpath: str,
        annotation_filepath: str,
        imageset_filepath: str,
        seg_label_dirpath: Optional[str]=None
    ) -> None:
        with open(annotation_filepath, "r") as f:
            annotation = json.loads(f.read())
        with open(imageset_filepath, "r") as f:
            for line in sorted(f.read().splitlines()):
                image_filepath = os.path.join(image_dirpath, line)
                cla_label = annotation["single_image"][line]["class"][0]  # TODO: need to handle soft-label and multi-label
                seg_label_filepath  = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
                self.append(
                    ImageAnnotationSingleton(
                        image=image_filepath,
                        cla_label=cla_label,
                        seg_label=seg_label_filepath
                    )
                )

    def from_directory_format(
        self,
        image_dirpaths: List[str],
        cla_labels: List[int],
        seg_label_dirpath: Optional[str]=None
    ) -> None:
        for image_dirpath, cla_label in zip(image_dirpaths, cla_labels):
            for ext in IMAGE_EXTS:
                for f in sorted(glob.glob(os.path.join(image_dirpath, f"*.{ext}"))):
                    fname = f.split(os.path.sep)[-1]
                    self.append(
                        ImageAnnotationSingleton(
                            image=f,
                            cla_label=cla_label,
                            seg_label=None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, fname)
                        )
                    )

class MultiImageAnnotation(list):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def from_research_format(
        self,
        image_dirpath: str,
        annotation_filepath: str,
        imageset_filepath: str,
        seg_label_dirpath: Optional[str]=None
    ) -> None:
        with open(annotation_filepath, "r") as f:
            annotation = json.loads(f.read())
        with open(imageset_filepath, "r") as f:
            for line in f.read().splitlines():
                product_dirpath = os.path.join(image_dirpath, line)
                image_filepaths = []
                for ext in IMAGE_EXTS:
                    image_filepaths += glob.glob(os.path.join(product_dirpath, f"*.{ext}"))
                cla_label = annotation["multi_image"][line]["class"][0]
                seg_label_filepath  = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
                self.append(
                    ImageAnnotationSingleton(
                        image=image_filepaths,
                        cla_label=cla_label,
                        seg_label=seg_label_filepath
                    )
                )
