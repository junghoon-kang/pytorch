import random
import numpy as np
import albumentations as A


__all__ = [
    'RandomCropNearDefect',
    'To3channel',
]


class RandomCropNearDefect(A.DualTransform):
    def __init__(self, size=(128,128), coverage_ratio=0.6, always_apply=False, p=1):
        super(RandomCropNearDefect, self).__init__(always_apply, p)
        if not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError('size should be list or tuple.')
        if len(size) != 2:
            raise ValueError('size should be list or tuple of size 2.')
        if not (isinstance(size[0], int) and isinstance(size[1], int)):
            raise TypeError('size should be list or tuple of integers.')
        if not (0 <= coverage_ratio <= 1):
            raise ValueError('coverage_ratio should be float between 0 and 1.')
        self.size = size
        self.coverage_ratio = coverage_ratio

    @property
    def targets_as_params(self):
        return ['image', 'mask']

    def get_params_dependent_on_targets(self, params):
        seg_label = params['mask']
        if np.sum(seg_label) == 0:
            h, w = seg_label.shape[:2]
            px = random.randint(self.size[0]//2, h - self.size[0]//2)
            py = random.randint(self.size[1]//2, h - self.size[1]//2)
            pivot = (px, py)  # middle pivot
        else:
            coverage_size = tuple(map(lambda l: int(l*self.coverage_ratio), self.size))
            coverage_pivot = tuple(map(lambda l: random.randint(0, l-1) if l != 0 else 0, coverage_size))
            dh = coverage_pivot[0] - (coverage_size[0] - 1) // 2
            dw = coverage_pivot[1] - (coverage_size[1] - 1) // 2
            indices = np.where(seg_label != 0)  # TODO: handle the case when there are more than one defect classes by random cropping near the specific defect class
            i = random.randint(0, len(indices[0])-1)
            defect_pivot = (indices[0][i], indices[1][i])
            pivot = (defect_pivot[0] - dh, defect_pivot[1] - dw)
        coords = self.get_coords_from_pivot(image, pivot, self.size)
        return {'coords': coords}

    def get_coords_from_pivot(self, image, pivot, size):
        h, w = image.shape[:2]
        if h < size[0] or w < size[1]:
            raise ValueError(f'Requested crop size ({size[0]}, {size[1]}) is larger than the image size ({h}, {w})')
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

    def apply(self, image, coords=(), **params):
        x_min, y_min, x_max, y_max = coords
        return A.functional.crop(image, x_min, y_min, x_max, y_max)

    def apply_to_mask(self, image, coords=(), **params):
        x_min, y_min, x_max, y_max = coords
        return A.functional.crop(image, x_min, y_min, x_max, y_max)

class To3channel(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super(To3channel, self).__init__(always_apply, p)

    def apply(self, image, **params):
        if len(image.shape) != 2:
            print(len(image.shape))
            raise ValueError('image should be 1-channel image')
        return np.dstack((image,)*3)


if __name__ == '__main__':
    import os, sys
    import skimage.io

    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(PATH, '..'))
    import config

    path = os.path.join(config.DATA_DIR, 'public', 'DAGM', 'original')
    names = [
        'domain1.test.NG.0002.png',
        'domain2.test.NG.0003.png',
        'domain3.test.NG.0007.png',
        'domain4.test.NG.0022.png',
        'domain5.test.NG.0001.png',
        'domain6.test.NG.0021.png',
        'domain7.test.NG.0005.png',
        'domain8.test.NG.0007.png',
        'domain9.test.NG.0009.png',
        'domain10.test.NG.0013.png'
    ]
    idx = 0
    image = skimage.io.imread( os.path.join(path, 'image', names[idx]) )
    seg_label = skimage.io.imread( os.path.join(path, 'mask', 'labeler.2class', names[idx]) )
    black = np.zeros((512,512), dtype=np.uint8)

    def RandomCropNearDefect_test():
        result = A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            RandomCropNearDefect(
                size = (128,128),
                coverage_ratio = 0.6
            )
        #])(image=image, mask=black)
        ])(image=image, mask=seg_label)
        x = result['image']
        y = result['mask']
        y[np.where(y == 0)] = 255
        y[np.where(y == 1)] = 0
        skimage.io.imsave('temp1.png', x)
        skimage.io.imsave('temp2.png', y)
        os.system('eog *.png')
        os.system('rm *.png')
    #RandomCropNearDefect_test()

    def To3channel_test():
        result = A.Compose([
            To3channel(),
        ])(image=image, mask=seg_label)
        x = result['image']
        y = result['mask']
        print(x.shape, y.shape)  # (512,512,3) (512,512)
    #To3channel_test()
