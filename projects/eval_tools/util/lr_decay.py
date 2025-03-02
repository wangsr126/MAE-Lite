# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# * https://github.com/microsoft/unilm/tree/master/beit
# * https://github.com/rwightman/pytorch-image-models/tree/master/timm
# * https://github.com/facebookresearch/deit
# * https://github.com/facebookresearch/dino
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# * https://github.com/rwightman/pytorch-image-models/tree/master/timm
# * https://github.com/facebookresearch/deit
# * https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False

from mae_lite.layers import LRSCHEDULERS
from mae_lite.layers.lr_scheduler import WarmCosineLRScheduler


@LRSCHEDULERS.register(name="warmcos_scale")
class WarmCosineWithScaleLRScheduler(WarmCosineLRScheduler):
    def step(self, count):
        count, inner_count = divmod(count, self.interval)
        if inner_count == 0:
            values = self.get_lr(count)
            for param_group, lr in zip(self.optimizer.param_groups, values):
                scale = param_group.get("lr_scale", 1.0)
                param_group["lr"] = lr * scale
            self.last_count = count


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    else:
        return num_max_layer - 1

def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

class LayerDecayValueAssigner_swin(object):
    def __init__(self, values, depths):
        self.values = values
        self.depths = depths

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_swin_layer(var_name, sum(self.depths), self.depths)

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.0
    else:
        # parameters = model.parameters()
        parameters = get_parameter_groups(model, 0.0, {}, get_num_layer, get_layer_scale)

    if "fused" in opt_lower:
        assert has_apex and torch.cuda.is_available(), "APEX and CUDA required for fused optimizers"

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps
    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas

    # print("optimizer settings:", opt_args)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "nadam":
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "sgdp":
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adafactor":
        if not args.lr:
            opt_args["lr"] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == "adahessian":
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == "rmsproptf":
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == "novograd":
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == "nvnovograd":
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == "fusedsgd":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "fusedmomentum":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == "fusedadam":
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == "fusedadamw":
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == "fusedlamb":
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == "fusednovograd":
        opt_args.setdefault("betas", (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer
