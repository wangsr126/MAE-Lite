# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# flake8: noqa F401, F403
from .cuda import Scaler, NativeScaler
from .checkpoint import save_checkpoint
from .env import collect_env_info, get_root_dir, find_free_port
from .log import setup_logger, setup_tensorboard_logger
from .misc import accuracy, AvgMeter, DataPrefetcher, DictAction, random_seed
from .registry import Registry


import torch
import torchvision

# PyTorch version as a tuple of 2 ints. Useful for comparison.
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
TORCHVISION_VERSION = tuple(int(x) for x in torchvision.__version__.split(".")[:2])
