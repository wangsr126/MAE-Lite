# flake8: noqa F401, F403

import timm
from .models_vit import *
from .registry import BACKBONES

__all__ = [k for k in globals().keys() if not k.startswith("_")]


def build_backbone(obj_type, *args, **kwargs):
    if obj_type in BACKBONES:
        return BACKBONES.get(obj_type)(*args, **kwargs)
    elif obj_type in timm.list_models():
        return timm.create_model(obj_type, *args, **kwargs)
    else:
        raise KeyError("No object named '{}' found in 'BACKBONES' registry!".format(obj_type))
