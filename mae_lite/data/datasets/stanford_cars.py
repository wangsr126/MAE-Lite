# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Modified from pytorch-fgvc-dataset (https://github.com/lvyilin/pytorch-fgvc-dataset)
# Copyright (c) 2020 lvyilin
# ---------------------------------------------------------
""" 
Standford Cars Dataset.
https://ai.stanford.edu/~jkrause/cars/car_dataset.html
"""
import os
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from ..registry import DATASETS
from mae_lite.utils import get_root_dir


@DATASETS.register()
class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        root (string): Root directory of the dataset.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    file_list = {
        "imgs": ("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz", "car_ims.tgz"),
        "annos": ("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat", "cars_annos.mat"),
    }

    def __init__(self, train=True, transform=None, target_transform=None, root=None, download=True):
        if root is None:
            root = os.path.join(get_root_dir(), "data/cars")
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.num_classes = 196
        self.loader = default_loader
        self.train = train

        if self._check_exists():
            print("Files already downloaded and verified.")
        elif download:
            self._download()
        else:
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list["annos"][1]))
        loaded_mat = loaded_mat["annotations"][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.file_list["imgs"][1])) and os.path.exists(
            os.path.join(self.root, self.file_list["annos"][1])
        )

    def _download(self):
        print("Downloading...")
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print("Extracting...")
        archive = os.path.join(self.root, self.file_list["imgs"][1])
        extract_archive(archive)


if __name__ == "__main__":
    train_dataset = Cars(train=True, download=True)
    test_dataset = Cars(train=False, download=True)
    print(len(train_dataset), len(test_dataset), train_dataset.num_classes)
