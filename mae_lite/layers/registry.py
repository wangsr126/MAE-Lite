# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
from mae_lite.utils.registry import Registry


LRSCHEDULERS = Registry("lr_scheduler")
OPTIMIZERS = Registry("optimizer")
