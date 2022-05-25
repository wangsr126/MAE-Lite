# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
import random
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image
from timm.data import create_transform as create_transform_timm
from .registry import TRANSFORMS


class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")


class Solarization(object):
    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@TRANSFORMS.register()
def ssl_transform(image_size=224):
    transform_q = transforms.Compose(
        [
            ToRGB(),
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_k = transforms.Compose(
        [
            ToRGB(),
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return [transform_q, transform_k]


@TRANSFORMS.register()
def typical_imagenet_transform(train, image_size=224, interpolation=Image.BILINEAR):
    if train:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(image_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        size = int((256 / 224) * image_size)
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
            ]
        )
    return transform


@TRANSFORMS.register()
def timm_transform(image_size, *args, **kwargs):
    resize_im = image_size > 32
    transform = transforms.Compose([ToRGB(), create_transform_timm(image_size, *args, **kwargs)])
    if not resize_im:
        # replace RandomResizedCropAndInterpolation with RandomCrop
        transform.transforms[1] = transforms.RandomCrop(image_size, padding=4)
    return transform
