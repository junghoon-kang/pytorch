import os, sys
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
    image = np.zeros((h,w), dtype=np.float32)
    label = np.zeros((h,w), dtype=np.uint8)
    return image, label

@pytest.fixture
def draw_rectangle(empty_image):
    image, label = empty_image
    h, w = image.shape[:2]
    random.seed(47)
    rec_center = (random.randint(0, h-1), random.randint(0, w-1))
    rec_size = (64,64)
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

@pytest.fixture
def draw_circle(empty_image):
    image, label = empty_image
    h, w = image.shape[:2]
    random.seed(47)
    cir_center = (random.randint(0, h-1), random.randint(0, w-1))
    cir_radius = 64
    x, y = np.ogrid[:h,:w]
    dist_from_center = np.sqrt((x - cir_center[0])**2 + (y - cir_center[1])**2)
    bool_indices = dist_from_center <= cir_radius
    image[bool_indices] = 255
    label[bool_indices] = 1
    return image, label

@pytest.fixture
def samples(draw_rectangle, draw_circle):
    rec_image, rec_label = draw_rectangle
    cir_image, cir_label = draw_circle
    images = [rec_image, cir_image]
    labels = [rec_label, cir_label]
    return images, labels

#def test_HorizontalFlip_1(samples):
#    for image, label in samples:
#        result = A.Compose([
#            A.HorizontalFlip(p=1)
#        ])(image=image)
#        out_image = result["image"]
#        assert np.array_equal(np.fliplr(image), out_image)

def test_HorizontalFlip_2(samples):
    #for image, label in samples:
    image, label = samples[1]
    result = A.Compose([
        A.HorizontalFlip(p=1)
    ])(image=image, mask=label)
    out_image = result["image"]
    out_label = result["mask"]
    imsave("out_image1.png", out_image)

    #label[np.where(label == 1)] = 255
    #out_label[np.where(out_label == 1)] = 255
    imsave("out_image2.png", out_image)

    #assert np.array_equal(image, label)
    #assert np.array_equal(out_image, out_label)
    assert np.array_equal(np.fliplr(image), out_image), str(np.max(image)) + "+" +  str(np.max(out_image))

#def test_HorizontalFlip_3(samples):
#    for image, label in samples:
#        result = A.Compose([
#            A.HorizontalFlip(p=1)
#        ])(image=image, seg_label=label)
#        out_image = result["image"]
#        out_label = result["seg_label"]
#
#        label[np.where(label == 1)] = 255
#        out_label[np.where(label == 1)] = 255
#        assert np.array_equal(image, label)
#        assert np.array_equal(image, out_label)
#        imsave("3.image.png", image)
#        imsave("3.out_image.png", out_image)
#        assert np.array_equal(np.fliplr(image), out_image)





if __name__ == "__main__":
    h, w = 512, 512
    image = np.zeros((h,w), dtype=np.float32)
    label = np.zeros((h,w), dtype=np.uint8)

    #rec_center = (random.randint(0, h-1), random.randint(0, w-1))
    #rec_size = (64,64)
    #t = rec_center[0] - int(np.floor(rec_size[0]/2))
    #b = rec_center[0] + int(np.ceil(rec_size[0]/2))
    #l = rec_center[1] - int(np.floor(rec_size[1]/2))
    #r = rec_center[1] + int(np.ceil(rec_size[1]/2))
    #if t < 0: t = 0
    #if b > h: b = h
    #if l < 0: l = 0
    #if r > w: r = w
    #indices = np.meshgrid(np.arange(t, b), np.arange(l, r), indexing='ij')
    #image[tuple(indices)] = 255
    #label[tuple(indices)] = 1

    cir_center = (random.randint(0, h-1), random.randint(0, w-1))
    cir_radius = 64
    x, y = np.ogrid[:h,:w]
    dist_from_center = np.sqrt((x - cir_center[0])**2 + (y - cir_center[1])**2)
    indices = dist_from_center <= cir_radius
    image[indices] = 255
    label[indices] = 1

    result = A.Compose([
        A.HorizontalFlip(p=1)
    ])(image=image, mask=label)
    out_image = result["image"]
    out_label = result["mask"]

    label[np.where(label == 1)] = 255
    out_label[np.where(label == 1)] = 255

    print(np.max(image), "+", np.max(out_image))

    #from IPython import embed; embed(); assert False


#@pytest.fixture
#def RandomCropNearDefect_answers(sample_names):
#    answer_dirpath = os.path.join(PATH, "test_transform", "RandomCropNearDefect")
#    images = [ imread(os.path.join(answer_dirpath, "image", name)) for name in sample_names ]
#    labels = [ imread(os.path.join(answer_dirpath, "label", name)) for name in sample_names ]
#    return zip(images, labels, sample_names)
#
#def test_RandomCropNearDefect(samples, RandomCropNearDefect_answers):
#    for (image, label, name), (answer_image, answer_label, answer_name) in zip(samples,RandomCropNearDefect_answers):
#        result = A.Compose([
#            RandomCropNearDefect(
#                size = (128,128),
#                coverage_size = (1,1),
#                fixed = True
#            )
#        ])(image=image, mask=label)
#
#        output_image = result["image"]
#        output_label = result["mask"]
#        output_label[np.where(output_label == 0)] = 255
#        output_label[np.where(output_label == 1)] = 0
#
#        assert name == answer_name
#        assert np.array_equal(output_image, answer_image)
#        assert np.array_equal(output_label, answer_label)
#
#def test_To3channel(samples):
#    for image, label, name in samples:
#        result = A.Compose([
#            To3channel(),
#        ])(image=image, mask=label)
#        output_image = result["image"]
#        output_label = result["mask"]
#        assert output_image.shape == (512,512,3)
#        assert output_label.shape == (512,512)


