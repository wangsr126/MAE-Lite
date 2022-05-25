# flake8: noqa F401, F403

from .lr_scheduler import *
from .optimizer import *
import torch.optim as optim

from .registry import LRSCHEDULERS, OPTIMIZERS


def build_lr_scheduler(obj_type, *args, **kwargs):
    return LRSCHEDULERS.get(obj_type)(*args, **kwargs)


def build_optimizer(obj_type, *args, **kwargs):
    opt_obj = getattr(optim, obj_type, None)
    if opt_obj is None:
        opt_obj = OPTIMIZERS.get(obj_type)
    return opt_obj(*args, **kwargs)
