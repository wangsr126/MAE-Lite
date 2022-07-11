# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Implementation from timm (https://github.com/rwightman/pytorch-image-models/tree/master/timm)
# Copyright 2020 Ross Wightman.
# --------------------------------------------------------
import torch


def dispatch_clip_grad(parameters, value: float, mode: str = "norm", norm_type: float = 2.0):
    """Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value'
        norm_type (float): p-norm, default 2.0
    """
    if mode == "norm":
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == "value":
        torch.nn.utils.clip_grad_value_(parameters, value)
    else:
        assert False, f"Unknown clip mode ({mode})."


class Scaler:
    """Naive Scaler"""

    def __init__(self):
        pass

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode="norm", parameters=None, create_graph=False, update_grad=True):
        loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                dispatch_clip_grad(parameters, value=clip_grad, mode=clip_mode)
            optimizer.step()

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode="norm", parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
