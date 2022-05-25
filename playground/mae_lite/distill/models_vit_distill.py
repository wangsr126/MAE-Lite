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
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import DropPath, Mlp, Attention as BaseAttn
from timm.models.registry import register_model


class Attention(BaseAttn):
    def __init__(self, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)

    def forward(self, x, with_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        ret_attn = attn if with_attn else None
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, ret_attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, with_hidden=False, with_attn=False):
        shortcut = x
        x = self.norm1(x)
        hidden = x if with_hidden else None
        x, attn = self.attn(x, with_attn=with_attn)
        x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, hidden, attn


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        global_pool=False,
        distilled=False,
        **kwargs
    ):
        super(VisionTransformer, self).__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **kwargs
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.num_heads = num_heads
        self.depth = depth
        self.global_pool = global_pool
        self.distilled = distilled
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
            if distilled:
                # distill norm & head
                self.dist_norm = norm_layer(embed_dim)
                if self.num_classes > 0:
                    self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
                else:
                    self.head_dist = nn.Identity()
        self.init_weights("")

    def forward_features(self, x, with_hidden=False, with_attn=False):
        # global_pool=True, distilled=True: (avg_token, dist_token)
        # global_pool=True, distilled=False:(avg_token, hiddens, attns)
        # global_pool=False,distilled=True: (cls_token, hiddens, attns)
        # global_pool=False,distilled=False:(cls_token, hiddens, attns)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hiddens = []
        attns = []
        for blk in self.blocks:
            x, hidden, attn = blk(x, with_hidden=with_hidden, with_attn=with_attn)
            hiddens.append(hidden)
            attns.append(attn)

        if self.global_pool:
            if self.distilled:
                x_dist, x = x[:, 0, :], x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x), self.dist_norm(x_dist)
            else:
                hiddens.append(self.fc_norm(x) if with_hidden else None)
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            hiddens.append(x if with_hidden else None)
            outcome = x[:, 0]

        attns = None if all([attn is None for attn in attns]) else attns
        hiddens = None if all([hidden is None for hidden in hiddens]) else hiddens
        return outcome, hiddens, attns


@register_model
def vit_tiny_patch16_distill(pretrained=False, **kwargs):
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
def vit_small_patch16_distill(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def vit_base_patch16_distill(pretrained=False, **kwargs):
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
def vit_large_patch16_distill(pretrained=False, **kwargs):
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
def vit_huge_patch14_distill(pretrained=False, **kwargs):
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
