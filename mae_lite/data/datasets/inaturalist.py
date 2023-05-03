import os
import os.path
from ..registry import DATASETS
from mae_lite.utils import get_root_dir

import json
from torchvision.datasets.folder import ImageFolder, default_loader


@DATASETS.register()
class INatDataset(ImageFolder):
    def __init__(self, train=True, root=None, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader, download=False):
        if root is None:
            root = os.path.join(get_root_dir(), "data/inaturalist")

        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.num_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, elem['file_name'])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


if __name__ == "__main__":
    train_dataset = INatDataset(train=True)
    test_dataset = INatDataset(train=False)
    print("Inatualist18: ", len(train_dataset), len(test_dataset), train_dataset.num_classes)
