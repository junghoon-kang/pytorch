import os, sys
import cv2
import pytest
import random
import numpy as np
import albumentations as A
from skimage.io import imread, imsave

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*2))
from vision.transform import *


@pytest.fixture
def empty_image():
    h, w = 512, 512
    image = np.full((h,w), fill_value=50, dtype=np.uint8)
    label = np.zeros((h,w), dtype=np.uint8)
    return image, label

def draw_rectangle(image, label, size):
    h, w = image.shape[:2]
    random.seed(47)
    rec_center = (random.randint(0, h-1), random.randint(0, w-1))
    rec_size = size
    t = rec_center[0] - int(np.floor(rec_size[0]/2))
    b = rec_center[0] + int(np.ceil(rec_size[0]/2))
    l = rec_center[1] - int(np.floor(rec_size[1]/2))
    r = rec_center[1] + int(np.ceil(rec_size[1]/2))
    if t < 0: t = 0
    if b > h: b = h
    if l < 0: l = 0
    if r > w: r = w
    indices = np.meshgrid(np.arange(t, b), np.arange(l, r), indexing='ij')
    image[tuple(indices)] = 255
    label[tuple(indices)] = 1
    return image, label

def draw_circle(image, label, size):
    h, w = image.shape[:2]
    random.seed(47)
    cir_center = (random.randint(0, h-1), random.randint(0, w-1))
    cir_radius = size
    x, y = np.ogrid[:h,:w]
    dist_from_center = np.sqrt((x - cir_center[0])**2 + (y - cir_center[1])**2)
    bool_indices = dist_from_center <= cir_radius
    image[bool_indices] = 255
    label[bool_indices] = 1
    return image, label

@pytest.fixture
def rectangle(empty_image):
    image, label = empty_image
    size = (64, 64)
    return draw_rectangle(image, label, size)

@pytest.fixture
def circle(empty_image):
    image, label = empty_image
    size = 64
    return draw_circle(image, label, size)

@pytest.fixture
def large_rectangle(empty_image):
    image, label = empty_image
    size = (128, 128)
    return draw_rectangle(image, label, size)

@pytest.fixture
def samples(rectangle, circle, large_rectangle):
    return [
        rectangle,
        circle,
        large_rectangle,
    ]


# albumentations.HorizontalFlip
def test_HorizontalFlip_1(samples):
    for image, label in samples:
        result = A.Compose([
            A.HorizontalFlip(p=1)
        ])(image=image)
        out_image = result["image"]

        assert np.array_equal(np.fliplr(image), out_image)

def test_HorizontalFlip_2(samples):
    for image, label in samples:
        result = A.Compose([
            A.HorizontalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.fliplr(image), out_image)
        assert np.array_equal(np.fliplr(label), out_label)
        image_y, image_x = np.where(image == 255)
        label_y, label_x = np.where(label == 1)
        assert np.array_equal(image_y, label_y) and np.array_equal(image_x, label_x)
        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

def test_HorizontalFlip_3(samples):
    for image, label in samples:
        result = A.Compose([
            A.HorizontalFlip(p=1)
        ])(image=image, seg_label=label)
        out_image = result["image"]
        out_label = result["seg_label"]

        assert np.array_equal(np.fliplr(image), out_image)
        assert np.array_equal(label, out_label)
        image_y, image_x = np.where(image == 255)
        label_y, label_x = np.where(label == 1)
        assert np.array_equal(image_y, label_y) and np.array_equal(image_x, label_x)

# albumentations.VerticalFlip
def test_VerticalFlip(samples):
    for image, label in samples:
        result = A.Compose([
            A.VerticalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.flipud(image), out_image)
        assert np.array_equal(np.flipud(label), out_label)
        image_y, image_x = np.where(image == 255)
        label_y, label_x = np.where(label == 1)
        assert np.array_equal(image_y, label_y) and np.array_equal(image_x, label_x)
        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

# Rotate90
def test_Rotate90(samples):
    for image, label in samples:
        result = A.Compose([
            Rotate90(p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.rot90(image, 1), out_image)
        assert np.array_equal(np.rot90(label, 1), out_label)
        image_y, image_x = np.where(image == 255)
        label_y, label_x = np.where(label == 1)
        assert np.array_equal(image_y, label_y) and np.array_equal(image_x, label_x)
        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

# albumentations.Rotate
def test_Rotate(samples):
    for image, label in samples:
        result = A.Compose([
            #A.Rotate(limit=(45,45), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            # Due to interpolation difference between image and label, transformed arrays may not match exactly.
            # - image: cv2.INTER_LINEAR
            # - label: cv2.INTER_NEAREST
            # For testing purpose, we use cv2.INTER_NEAREST interpolation for image as well.
            A.Rotate(limit=(45,45), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        image_y, image_x = np.where(image == 255)
        label_y, label_x = np.where(label == 1)
        assert np.array_equal(image_y, label_y) and np.array_equal(image_x, label_x)
        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

# albumentations.RandomBrightnessContrast
def test_RandomBrightnessContrast_1(samples):
    for image, label in samples:
        result = A.Compose([
            A.RandomBrightnessContrast(contrast_limit=(1,1), brightness_limit=0)  # Contrast
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(label, out_label)
        ok = np.where(label == 0)
        i, j = ok[0][0], ok[0][1]
        assert np.clip(image[i,j]*2, a_min=0, a_max=255) == out_image[i,j]
        ng = np.where(label == 1)
        i, j = ng[0][0], ng[0][1]
        assert np.clip(image[i,j]*2, a_min=0, a_max=255) == out_image[i,j]

def test_RandomBrightnessContrast_2(samples):
    for image, label in samples:
        result = A.Compose([
            A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=(.5,.5))  # Brightness
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(label, out_label)
        ok = np.where(label == 0)
        i, j = ok[0][0], ok[0][1]
        assert np.clip(image[i,j] + int(255/2), a_min=0, a_max=255) == out_image[i,j]
        ng = np.where(label == 1)
        i, j = ng[0][0], ng[0][1]
        assert np.clip(image[i,j] + int(255/2), a_min=0, a_max=255) == out_image[i,j]

# ZoomIn
def test_ZoomIn(samples):
    for image, label in samples:
        result = A.Compose([
            #ZoomIn(scale_limit=(1.2,1.2), interpolation=cv2.INTER_LINEAR, p=1)
            # Due to interpolation difference between image and label, transformed arrays may not match exactly.
            # - image: cv2.INTER_LINEAR
            # - label: cv2.INTER_NEAREST
            # For testing purpose, we use cv2.INTER_NEAREST interpolation for image as well.
            ZoomIn(scale_limit=(1.2,1.2), interpolation=cv2.INTER_NEAREST, p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert image.shape == out_image.shape
        assert label.shape == out_label.shape
        assert not np.array_equal(label, out_label)
        image_y, image_x = np.where(image == 255)
        label_y, label_x = np.where(label == 1)
        assert np.array_equal(image_y, label_y) and np.array_equal(image_x, label_x)
        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

# albumentations.Sharpen
#def test_Sharpen(samples):
#    for image, label in samples:
#        result = A.Compose([
#            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
#        ])(image=image, mask=label)
#        out_image = result["image"]
#        out_label = result["mask"]
#
#        assert np.array_equal(label, out_label)
#        # TODO: write more test cases

# albumentations.GaussianBlur
def test_GaussianBlur(samples):
    for image, label in samples:
        result = A.Compose([
            A.GaussianBlur(blur_limit=(3,3), sigma_limit=(0,2))
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(label, out_label)
        # TODO: write more test cases

# albumentations.MultiplicativeNoise
def test_MultiplicativeNoise(samples):
    for image, label in samples:
        result = A.Compose([
            A.MultiplicativeNoise(multiplier=(0,2), per_channel=False, elementwise=True)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(label, out_label)
        # TODO: write more test cases

# RandomCropNearDefect
def test_RandomCropNearDefect(samples):
    for image, label in samples:
        result = A.Compose([
            RandomCropNearDefect(
                size = (128,128),
                coverage_size = (1,1),
                fixed = True
            )
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert out_image.shape == (128,128)
        assert out_label.shape == (128,128)
        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

# To3channel
def test_To3channel(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert out_image.shape == (512,512,3)
        assert out_label.shape == (512,512)


if __name__ == "__main__":
    h, w = 512, 512
    image = np.full((h,w), fill_value=50, dtype=np.uint8)
    label = np.zeros((h,w), dtype=np.uint8)
    random.seed(47)
    rec_center = (random.randint(0, h-1), random.randint(0, w-1))
    rec_size = (128,128)
    t = rec_center[0] - int(np.floor(rec_size[0]/2))
    b = rec_center[0] + int(np.ceil(rec_size[0]/2))
    l = rec_center[1] - int(np.floor(rec_size[1]/2))
    r = rec_center[1] + int(np.ceil(rec_size[1]/2))
    if t < 0: t = 0
    if b > h: b = h
    if l < 0: l = 0
    if r > w: r = w
    indices = np.meshgrid(np.arange(t, b), np.arange(l, r), indexing='ij')
    image[tuple(indices)] = 255
    label[tuple(indices)] = 1

    result = A.Compose([
        ZoomIn(scale_limit=(1.2,1.2), interpolation=cv2.INTER_LINEAR, p=1),
        A.GaussianBlur(blur_limit=(3,3), sigma_limit=(0,2), p=1),
        A.MultiplicativeNoise(multiplier=(0,2), per_channel=False, elementwise=True, p=1),
    ])(image=image, mask=label)

    out_image = result["image"]
    out_label = result["mask"]
    label[np.where(label==1)] = 255
    out_label[np.where(out_label==1)] = 255
    imsave("image.png", image)
    imsave("label.png", label)
    imsave("out_image.png", out_image)
    imsave("out_label.png", out_label)
    #from IPython import embed; embed(); assert False
