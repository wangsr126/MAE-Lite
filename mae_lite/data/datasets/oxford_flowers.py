# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Modified from MoCo-v3 (https://github.com/facebookresearch/moco-v3)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# ---------------------------------------------------------
""" 
Oxford Flowers Datasets
https://www.robots.ox.ac.uk/~vgg/data/flowers/
"""
from __future__ import print_function
from PIL import Image

import numpy as np
import os
import os.path
import scipy.io

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from ..registry import DATASETS
from mae_lite.utils import get_root_dir


@DATASETS.register()
class Flowers(VisionDataset):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
        root=None,
        download=True,
    ):
        if root is None:
            root = os.path.join(get_root_dir(), "data/flowers")
        super(Flowers, self).__init__(root, transform=transform, target_transform=target_transform)

        base_folder = root
        self.num_classes = 102
        self.image_folder = os.path.join(base_folder, "jpg")
        self.label_file = os.path.join(base_folder, "imagelabels.mat")
        self.setid_file = os.path.join(base_folder, "setid.mat")

        self.url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        self.url_labels = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
        self.url_setids = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

        if download:
            self.download()

        self.train = train

        self.labels = scipy.io.loadmat(self.label_file)["labels"][0]
        train_list = scipy.io.loadmat(self.setid_file)["trnid"][0]
        val_list = scipy.io.loadmat(self.setid_file)["valid"][0]
        test_list = scipy.io.loadmat(self.setid_file)["tstid"][0]
        trainval_list = np.concatenate([train_list, val_list])

        if self.train:
            self.img_files = trainval_list
        else:
            self.img_files = test_list

    def __getitem__(self, index):
        img_name = "image_%05d.jpg" % self.img_files[index]
        target = self.labels[self.img_files[index] - 1] - 1
        img = Image.open(os.path.join(self.image_folder, img_name))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_files)

    def _check_exists(self):
        return os.path.exists(self.image_folder) and os.path.exists(self.label_file) and os.path.exists(self.setid_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data from "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        print("Downloading %s..." % self.url)
        tar_name = self.url.rpartition("/")[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print("Extracting %s..." % tar_path)
        extract_archive(tar_path)
        print("Downloading %s..." % self.url_labels)
        download_url(self.url_labels, root=self.root, filename=self.url_labels.rpartition("/")[-1])
        print("Downloading %s..." % self.url_setids)
        download_url(self.url_setids, root=self.root, filename=self.url_setids.rpartition("/")[-1])
        print("Done!")


if __name__ == "__main__":
    train_dataset = Flowers(train=True, download=True)
    test_dataset = Flowers(train=False, download=True)
    print(len(train_dataset), len(test_dataset), train_dataset.num_classes)
