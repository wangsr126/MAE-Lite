# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
""" 
CIFAR10/100 Datasets. 
https://www.cs.toronto.edu/~kriz/cifar.html
"""
import os.path as osp
import torchvision.datasets as datasets
from mae_lite.utils import get_root_dir
from ..registry import DATASETS


@DATASETS.register()
class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, transform=None):
        root = osp.join(get_root_dir(), "data/cifar")
        super().__init__(root, train, transform=transform, target_transform=None, download=True)
        self.num_classes = 10


@DATASETS.register()
class CIFAR100(datasets.CIFAR100):
    def __init__(self, train, transform=None):
        root = osp.join(get_root_dir(), "data/cifar")
        super().__init__(root, train, transform=transform, target_transform=None, download=True)
        self.num_classes = 100


if __name__ == "__main__":
    train_dataset = CIFAR10(train=True, download=True)
    test_dataset = CIFAR10(train=False, download=True)
    print("CIFAR10:", len(train_dataset), len(test_dataset), train_dataset.num_classes)
    train_dataset = CIFAR100(train=True, download=True)
    test_dataset = CIFAR100(train=False, download=True)
    print(len(train_dataset), len(test_dataset), train_dataset.num_classes)
