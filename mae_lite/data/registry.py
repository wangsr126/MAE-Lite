# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
from mae_lite.utils.registry import Registry


DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
