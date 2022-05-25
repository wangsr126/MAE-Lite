# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
""" 
ImageNet Datasets. 
https://www.image-net.org/
"""
from torchvision.datasets import ImageFolder
from ..registry import DATASETS


@DATASETS.register()
class SSL_ImageNet(ImageFolder):
    def __init__(self, transform=None, root="data/imagenet/imagenet_train", target_transform=None, is_valid_file=None):
        super(SSL_ImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file
        )
        # self.image_dir + f"imagenet_{'train' if stage in ('train', 'ft') else 'val'}"

        if transform is not None:
            if isinstance(transform, list) and len(transform) > 1:
                self.transform, self.transform_k = transform
            else:
                self.transform, self.transform_k = transform, None
        else:
            raise ValueError("Transform function missing!")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        img1 = self.transform(sample)
        if self.transform_k is not None:
            img2 = self.transform_k(sample)
        else:
            img2 = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target


@DATASETS.register()
class ImageNet(ImageFolder):
    def __init__(self, train, transform=None):
        root = "data/imagenet/imagenet_{}".format("train" if train else "val")
        super(ImageNet, self).__init__(root, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
