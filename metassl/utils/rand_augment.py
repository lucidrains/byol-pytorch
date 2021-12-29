# Code based on:  https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# Code adapted to work also for segmentation
import random
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps

from albumentations import CenterCrop
from albumentations import Compose
from albumentations import HorizontalFlip
from albumentations import PadIfNeeded
from albumentations import RandomScale
from albumentations import Rotate

########################################################################################################################
# IDENTITY
########################################################################################################################

def Identity(data, _, __):
    return data


########################################################################################################################
# COLOR OPS
########################################################################################################################

def AutoContrast(data, v, is_segmentation):
    if is_segmentation:
        return PIL.ImageOps.autocontrast(data[0], v), data[1]
    else:
        return PIL.ImageOps.autocontrast(data, v)


def Invert(data, _, is_segmentation):
    if is_segmentation:
        return PIL.ImageOps.invert(data[0]), data[1]
    else:
        return PIL.ImageOps.invert(data)


def Equalize(data, _, is_segmentation):
    if is_segmentation:
        return PIL.ImageOps.equalize(data[0]), data[1]
    else:
        return PIL.ImageOps.equalize(data)


def Solarize(data, v, is_segmentation):  # [0, 256]
    assert 0 <= v <= 256
    if is_segmentation:
        return PIL.ImageOps.solarize(data[0], v), data[1]
    else:
        return PIL.ImageOps.solarize(data, v)


def Posterize(data, v, is_segmentation):  # [4, 8]
    v = int(v)
    v = max(1, v)
    if is_segmentation:
        return PIL.ImageOps.posterize(data[0], v), data[1]
    else:
        return PIL.ImageOps.posterize(data, v)


def Contrast(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Contrast(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Contrast(data).enhance(v)


def Color(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Color(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Color(data).enhance(v)


def Brightness(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Brightness(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Brightness(data).enhance(v)


def Sharpness(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Sharpness(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Sharpness(data).enhance(v)


########################################################################################################################
# GEOMETRIC OPS
########################################################################################################################

def ShearX(data, v, is_segmentation):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), PIL.Image.NEAREST
        )
        return image, mask
    else:
        return data.transform(data[0].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), PIL.Image.BILINEAR)


def ShearY(data, v, is_segmentation):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), PIL.Image.NEAREST
        )
        return image, mask
    else:
        return data.transform(data.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), PIL.Image.BILINEAR)


def TranslateX(data, v, is_segmentation):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        v_0 = v * data[0].size[0]
        v_1 = v * data[1].size[0]
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, 0, v_0, 0, 1, 0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, 0, v_1, 0, 1, 0), PIL.Image.NEAREST
        )
        return image, mask
    else:
        v = v * data.size[0]
        return data.transform(data.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), PIL.Image.BILINEAR)


def TranslateY(data, v, is_segmentation):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        v_0 = v * data[0].size[0]
        v_1 = v * data[1].size[0]
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v_0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v_1), PIL.Image.NEAREST
        )
        return image, mask
    else:
        v = v * data.size[0]
        return data.transform(data.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), PIL.Image.BILINEAR)


def Rotate(data, v, is_segmentation):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        return data[0].rotate(v), data[1].rotate(v)
    else:
        return data.rotate(v)

########################################################################################################################

def augment_list():  # default opterations used in RandAugment paper
    augment_list = [
        (Identity, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (Rotate, 0, 30),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (TranslateX, 0.0, 0.33),
        (TranslateY, 0.0, 0.33),
    ]

    return augment_list


class RandAugmentDefaultOps:
    def __init__(self, num_ops, magnitude, is_segmentation=False):
        self.num_ops = num_ops  # TODO: optimize with BOHB
        self.magnitude = magnitude  # TODO: optimize with BOHB
        self.augment_list = augment_list()
        self.is_segmentation = is_segmentation

    def __call__(self, data):
        ops = random.choices(self.augment_list, k=self.num_ops)
        for op, minval, maxval in ops:
            magnitude_val = (float(self.magnitude) / 30) * float(maxval - minval) + minval
            data = op(data, magnitude_val, self.is_segmentation)

        return data