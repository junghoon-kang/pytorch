import os
import glob
import json
from typing import Optional, List


IMAGE_EXTS = [
    "bmp",
    "png",
    "jpg",
    "jpeg",
]

class ImageAnnotation(list):
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
                image_filepath = os.path.join(image_dirpath, line)
                cla_label_list = annotation["single_image"][line]["class"]
                seg_label_filepath  = None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, line)
                self.append(
                    ImageAnnotationSingleton(
                        image=image_filepath,
                        cla_label=cla_label_list,
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
                for f in glob.glob(os.path.join(image_dirpath, f"*.{ext}")):
                    fname = f.split(os.path.sep)[-1]
                    self.append(
                        ImageAnnotationSingleton(
                            image=f,
                            cla_label=[cla_label],
                            seg_label=None if seg_label_dirpath is None else os.path.join(seg_label_dirpath, fname)
                        )
                    )

class ImageAnnotationSingleton(object):
    def __init__(
        self,
        image: str,
        cla_label: Optional[List[int]]=None,
        seg_label: Optional[str]=None
    ):
        self._image = image
        self._cla_label = cla_label
        self._seg_label = seg_label

    def __repr__(self):
        return f"Object(image='{self.image}', cla_label='{self.cla_label}', seg_label='{self.seg_label}')"

    @property
    def image(self):
        return self._image

    @property
    def seg_label(self):
        return self._seg_label

    @property
    def cla_label(self):
        return self._cla_label
