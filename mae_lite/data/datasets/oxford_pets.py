# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Modified from MoCo-v3 (https://github.com/facebookresearch/moco-v3)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# ---------------------------------------------------------
""" 
Oxford Pets Datasets
https://www.robots.ox.ac.uk/~vgg/data/pets/
"""
from PIL import Image
from typing import Any, Callable, Optional, Tuple

import os
import os.path
from ..registry import DATASETS
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from mae_lite.utils import get_root_dir


@DATASETS.register()
class Pets(VisionDataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        root: str = None,
        download: bool = True,
    ) -> None:
        if root is None:
            root = os.path.join(get_root_dir(), "data/pets")
        super(Pets, self).__init__(root, transform=transform, target_transform=target_transform)

        base_folder = root
        self.train = train
        self.num_classes = 37
        self.annotations_path_dir = os.path.join(base_folder, "annotations")
        self.image_path_dir = os.path.join(base_folder, "images")

        self.url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        self.url_labels = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

        if download:
            self.download()

        if self.train:
            split_file = os.path.join(self.annotations_path_dir, "trainval.txt")
            with open(split_file) as f:
                self.images_list = f.readlines()
        else:
            split_file = os.path.join(self.annotations_path_dir, "test.txt")
            with open(split_file) as f:
                self.images_list = f.readlines()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_name, label, species, _ = self.images_list[index].strip().split(" ")

        img_name += ".jpg"
        target = int(label) - 1

        img = Image.open(os.path.join(self.image_path_dir, img_name))
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.images_list)

    def _check_exists(self):
        return os.path.exists(self.annotations_path_dir) and os.path.exists(self.image_path_dir)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data
        print("Downloading %s..." % self.url)
        tar_name = self.url.rpartition("/")[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print("Extracting %s..." % tar_path)
        extract_archive(tar_path)
        print("Downloading %s..." % self.url_labels)
        tar_label_name = self.url_labels.rpartition("/")[-1]
        download_url(self.url_labels, root=self.root, filename=tar_label_name)
        tar_label_path = os.path.join(self.root, tar_label_name)
        print("Extracting %s..." % tar_label_path)
        extract_archive(tar_label_path)
        print("Done!")


if __name__ == "__main__":
    train_dataset = Pets(train=True, download=True)
    test_dataset = Pets(train=False, download=True)
    print(len(train_dataset), len(test_dataset), train_dataset.num_classes)
