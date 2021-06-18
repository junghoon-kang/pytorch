import os, sys
import cv2
import torch
import scipy
import pytest
import random
import numpy as np
import albumentations as A

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, *[".."]*3))
from vision.transform import *


@pytest.fixture
def empty_image():
    h, w = 512, 512
    image = np.full((h,w), fill_value=50, dtype=np.uint8)
    label = np.zeros((h,w), dtype=np.uint8)
    return (image, label)

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
    return (image, label)

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
    return (image, label)

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

####################################################################################################
# ToTensor
def test_ToTensor_1(samples):
    for image, label in samples:
        result = A.Compose([
            ToTensor(),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert out_image.shape == torch.Size([1,512,512])
        assert out_label.shape == torch.Size([512,512])

def test_ToTensor_2(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
            ToTensor(),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert out_image.shape == torch.Size([3,512,512])
        assert out_label.shape == torch.Size([512,512])

####################################################################################################
# To3channel
def test_To3channel_1(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert out_image.shape == (512,512,3)
        assert out_label.shape == (512,512)

def test_To3channel_2(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert image.dtype == out_image.dtype, image.dtype
        assert label.dtype == out_label.dtype, label.dtype

def test_To3channel_3(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(out_image[:,:,0], out_image[:,:,1])
        assert np.array_equal(out_image[:,:,1], out_image[:,:,2])

####################################################################################################
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

def test_HorizontalFlip_3(samples):
    for image, label in samples:
        result = A.Compose([
            A.HorizontalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

def test_HorizontalFlip_4(samples):
    for image, label in samples:
        result = A.Compose([
            A.HorizontalFlip(p=1)
        ])(image=image, seg_label=label)
        out_image = result["image"]
        out_label = result["seg_label"]

        assert np.array_equal(np.fliplr(image), out_image)
        assert np.array_equal(label, out_label)

def test_HorizontalFlip_5(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
            A.HorizontalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.fliplr(image), out_image[:,:,0])
        assert np.array_equal(np.fliplr(label), out_label)

####################################################################################################
# albumentations.VerticalFlip
def test_VerticalFlip_1(samples):
    for image, label in samples:
        result = A.Compose([
            A.VerticalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.flipud(image), out_image)
        assert np.array_equal(np.flipud(label), out_label)

def test_VerticalFlip_2(samples):
    for image, label in samples:
        result = A.Compose([
            A.VerticalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

def test_VerticalFlip_3(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
            A.VerticalFlip(p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.flipud(image), out_image[:,:,0])
        assert np.array_equal(np.flipud(label), out_label)

####################################################################################################
# Rotate90
def test_Rotate90_1(samples):
    for image, label in samples:
        result = A.Compose([
            Rotate90(p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.rot90(image, 1), out_image)
        assert np.array_equal(np.rot90(label, 1), out_label)

def test_Rotate90_2(samples):
    for image, label in samples:
        result = A.Compose([
            Rotate90(p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

def test_Rotate90_3(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
            Rotate90(p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(np.rot90(image, 1), out_image[:,:,0])
        assert np.array_equal(np.rot90(label, 1), out_label)

####################################################################################################
# albumentations.Rotate
def apply_rotation(image, angle, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0):
    h, w = image.shape[:2]
    center = w//2, h//2
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    result = cv2.warpAffine(image, M=matrix, dsize=(w, h), flags=interpolation, borderMode=border_mode, borderValue=value)
    return result

def test_Rotate_1(samples):
    for image, label in samples:
        result = A.Compose([
            A.Rotate(limit=(45,45), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(apply_rotation(image, 45), out_image)
        assert np.array_equal(apply_rotation(label, 45), out_label)

def test_Rotate_2(samples):
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

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

def test_Rotate_3(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
            A.Rotate(limit=(45,45), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(apply_rotation(image, 45), out_image[:,:,0])
        assert np.array_equal(apply_rotation(label, 45), out_label)

####################################################################################################
# albumentations.RandomBrightnessContrast
def apply_contrast(image, alpha=1):
    alpha += 1
    lut = np.arange(0, 255+1).astype(np.float32)
    if alpha != 1:
        lut *= alpha
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    result = cv2.LUT(image, lut)
    return result

def test_Contrast_1(samples):
    for image, label in samples:
        result = A.Compose([
            A.RandomBrightnessContrast(contrast_limit=(1,1), brightness_limit=0)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(apply_contrast(image, 1), out_image)
        assert np.array_equal(label, out_label)

def test_Contrast_2(samples):
    for image, label in samples:
        result = A.Compose([
            A.RandomBrightnessContrast(contrast_limit=(1,1), brightness_limit=0)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

# TODO: need to check why it is not working
#def test_Contrast_3_(samples):
#    for image, label in samples:
#        result = A.Compose([
#            To3channel(),
#            A.RandomBrightnessContrast(contrast_limit=(1,1), brightness_limit=0)
#        ])(image=image, mask=label)
#        out_image = result["image"]
#        out_label = result["mask"]
#
#        assert np.array_equal(apply_contrast(image, 2), out_image[:,:,0]), f"\n\n{np.histogram(apply_contrast(image, 2))}\n\n{np.histogram(out_image[:,:,0])}\n\n"
#        assert np.array_equal(label, out_label)

@pytest.mark.parametrize("i", range(3))
def test_Contrast_3(samples, i):
    image, label = samples[i]
    result = A.Compose([
        To3channel(),
        A.RandomBrightnessContrast(contrast_limit=(1,1), brightness_limit=0)
    ])(image=image, mask=label)
    out_image = result["image"]
    out_label = result["mask"]

    assert np.array_equal(apply_contrast(image, 1), out_image[:,:,0]), f"\n\n{np.histogram(apply_contrast(image, 2))}\n\n{np.histogram(out_image[:,:,0])}\n\n"
    assert np.array_equal(label, out_label)

def apply_brightness(image, beta=0.5):
    lut = np.arange(0, 255+1).astype(np.float32)
    if beta != 0:
        lut += beta * 255
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    result = cv2.LUT(image, lut)
    return result

def test_Brightness_1(samples):
    for image, label in samples:
        result = A.Compose([
            A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=(.5,.5))
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(apply_brightness(image, 0.5), out_image)
        assert np.array_equal(label, out_label)

def test_Brightness_2(samples):
    for image, label in samples:
        result = A.Compose([
            A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=(.5,.5))
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

@pytest.mark.parametrize("i", range(3))
def test_Brightness_3(samples, i):
    image, label = samples[i]
    result = A.Compose([
    To3channel(),
        A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=(.5,.5))
    ])(image=image, mask=label)
    out_image = result["image"]
    out_label = result["mask"]

    assert np.array_equal(apply_brightness(image, 0.5), out_image[:,:,0]), f"\n\n{np.histogram(apply_brightness(image, 0.5))}\n\n{np.histogram(out_image[:,:,0])}\n\n"
    assert np.array_equal(label, out_label)

####################################################################################################
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

####################################################################################################
# Sharpen
def test_Sharpen(samples):
    for image, label in samples:
        result = A.Compose([
            Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(label, out_label)
        # TODO: write more test cases

####################################################################################################
# albumentations.GaussianBlur
def test_GaussianBlur(samples):
    for image, label in samples:
        result = A.Compose([
            A.GaussianBlur(blur_limit=(5,5), sigma_limit=(0,2))
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        assert np.array_equal(label, out_label)
        # TODO: write more test cases

####################################################################################################
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

####################################################################################################
# RandomCropNearDefect
def test_RandomCropNearDefect_1(samples):
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

def test_RandomCropNearDefect_2(samples):
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

        out_image_y, out_image_x = np.where(out_image == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)

def test_RandomCropNearDefect_3(samples):
    for image, label in samples:
        result = A.Compose([
            To3channel(),
            RandomCropNearDefect(
                size = (128,128),
                coverage_size = (1,1),
                fixed = True
            )
        ])(image=image, mask=label)
        out_image = result["image"]
        out_label = result["mask"]

        out_image_y, out_image_x = np.where(out_image[:,:,0] == 255)
        out_label_y, out_label_x = np.where(out_label == 1)
        assert np.array_equal(out_image_y, out_label_y) and np.array_equal(out_image_x, out_label_x)


if __name__ == "__main__":
    h, w = 512, 512
    image = np.full((h,w), fill_value=50, dtype=np.uint8)
    label = np.zeros((h,w), dtype=np.uint8)
    random.seed(47)
    #image, label = draw_rectangle(image, label, (128,128))
    image, label = draw_rectangle(image, label, (64,64))

    result = A.Compose([
        To3channel(),
        A.RandomBrightnessContrast(contrast_limit=(1,1), brightness_limit=0)
    ])(image=image, mask=label)
    out_image = result["image"]
    out_label = result["mask"]

    answer = apply_contrast(image, 2)

    print(
        np.array_equal(answer, out_image[:,:,0])
    )

    #from skimage.io import imsave
    #label[np.where(label==1)] = 255
    #out_label[np.where(out_label==1)] = 255
    #imsave("image.png", image)
    #imsave("label.png", label)
    #imsave("out_image.png", out_image)
    #imsave("out_label.png", out_label)

    from IPython import embed; embed(); assert False
