# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
""" 
Vision Transformer with GAP
"""

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.registry import register_model


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling and distillation"""

    def __init__(self, global_pool=False, distilled=False, **kwargs):
        super(VisionTransformer, self).__init__(distilled=False if global_pool else distilled, **kwargs)

        self.global_pool = global_pool
        self.distilled = distilled
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
            if self.distilled:
                self.dist_norm = norm_layer(embed_dim)
                if self.num_classes > 0:
                    self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
                else:
                    self.head_dist = nn.Identity()
        self.init_weights("")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # if GAP, then this cls_tokens is used as distill_tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            if self.distilled:
                x_dist, x = x[:, 0, :], x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x), self.dist_norm(x_dist)
            else:
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                return self.fc_norm(x)
        else:
            x = self.norm(x)
            return x[:, 0]


@register_model
def vit_tiny_patch16(pretrained=False, **kwargs):
    # the number of heads is changed to 12 from 3, which is different to the original arch.
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def vit_base_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def vit_large_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def vit_huge_patch14(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
