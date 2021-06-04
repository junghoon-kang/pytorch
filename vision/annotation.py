class ImageAnnotation:
    def __init__(self, image):
        pass

class ImageAnnotationSingleton(object):
    def __init__(self, image: str, cla_label: int=None, seg_label: str=None):
        self._image = image
        self._cla_label = cla_label
        self._seg_label = seg_label

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label
