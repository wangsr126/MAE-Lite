# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
# flake8: noqa F401, F403
from .datasets.imagenet import SSL_ImageNet, ImageNet
from .datasets.cifar import CIFAR10, CIFAR100
from .datasets.fgvc_aircraft import Aircraft
from .datasets.inaturalist import INatDataset
from .datasets.oxford_flowers import Flowers
from .datasets.oxford_pets import Pets
from .datasets.stanford_cars import Cars

from .transforms import ssl_transform, typical_imagenet_transform, timm_transform
from .registry import DATASETS
from .registry import TRANSFORMS


__all__ = [k for k in globals().keys() if not k.startswith("_")]


def build_dataset(obj_type, *args, **kwargs):
    return DATASETS.get(obj_type)(*args, **kwargs)


def build_transform(obj_type, *args, **kwargs):
    return TRANSFORMS.get(obj_type)(*args, **kwargs)
