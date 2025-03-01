# --------------------------------------------------------
# Copyright (c) 2025 Institute of Automation Chinese Academy of Sciences. All Rights Reserved.
# --------------------------------------------------------
# Modified from SimMIM (https://github.com/microsoft/SimMIM)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on BEIT code bases (https://github.com/microsoft/unilm/tree/master/beit)
# Written by Yutong Lin, Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from swin_transformer import SwinTransformer, SwinTransformerBlock
from timm.models import register_model
# from .vision_transformer import VisionTransformer


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, decoder_idx=None, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.decoder_idx = decoder_idx

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, flatten=False):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if len(self.decoder_idx) > 1:
            interm_list = []
        for i, layer in enumerate(self.layers):
            x_ori, x = layer(x)
            if len(self.decoder_idx) > 1:
                if i in self.decoder_idx:
                    if x_ori != None:
                        interm_list.append(x_ori)
                    else:
                        interm_list.append(x)
            else:
                if i in self.decoder_idx:
                    x_dec = x
        if len(self.decoder_idx) > 1:
            return interm_list
        
        x_dec = self.norm(x_dec)
        if not flatten:
            x_dec = x_dec.transpose(1, 2)
            B, C, L = x_dec.shape
            H = W = int(L ** 0.5)
            x_dec = x_dec.reshape(B, C, H, W)
        return x_dec

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, if_stu=False, decoder_embed_dim=96):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        if if_stu:
            self.decoder = nn.Sequential(
                nn.Linear(self.encoder.num_features // max(1, 2 ** (2-self.encoder.decoder_idx[0])), decoder_embed_dim),
                SwinTransformerBlock(dim=decoder_embed_dim,
                                    input_resolution=(self.encoder.patches_resolution[0] // (2 ** min(3,(self.encoder.decoder_idx[0]+1))),
                                                    self.encoder.patches_resolution[1] // (2 **  min(3,(self.encoder.decoder_idx[0]+1)))),
                                    num_heads=3),
            )
            self.decoder_pred = nn.Sequential(
                nn.Conv2d(
                    in_channels=decoder_embed_dim,
                    out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )
            self.decoder.apply(self._init_weights)
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )
            

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.if_stu = if_stu
        if if_stu:
            del self.encoder.norm
            self.encoder.norm = nn.LayerNorm(self.encoder.num_features // max(1, 2 ** (2-self.encoder.decoder_idx[0])))
            if len(self.encoder.layers[3].blocks) > 2:
                del self.encoder.layers[3].blocks[2].mlp
                del self.encoder.layers[3].blocks[2].norm2
                del self.encoder.layers[3].blocks[2].attn.proj

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask):
        return self.encoder(x, mask, self.if_stu)

    def forward(self, x, mask):
        # z = self.encoder(x, mask)
        z = self.forward_encoder(x, mask)
        # if self.if_stu:
        #     x_fuse = 0.
        #     for x_p, fuse in zip(interm_list, self.fuse_module):
        #         x_fuse = x_fuse + fuse(patchify(x_p))
        #     z = self.encoder.norm(unpatchify(x_fuse))
        # else:
        #     z = interm_list
        x_rec = self.decoder(z)
        if self.if_stu:
            x_rec = x_rec.transpose(1, 2)
            B, C, L = x_rec.shape
            H = W = int(L ** 0.5)
            x_rec = x_rec.reshape(B, C, H, W)
            x_rec = self.decoder_pred(x_rec)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def patchify(x):
    x = x.transpose(1, 2)
    B, C, L = x.shape
    H = W = int(L ** 0.5)
    x = x.reshape(B, C, H, W)
    return x

def unpatchify(x):
    x = x.flatten(2).transpose(1, 2)
    return x

@register_model
def simmim_swin_tiny(pretrained=False, **kwargs):
    encoder = SwinTransformerForSimMIM(
        img_size = 224,
        embed_dim = 48,
        window_size = 7,
        depths=[2, 2, 6, 2], 
        num_heads=[ 3, 6, 12, 24 ],
        num_classes=0,
        decoder_idx=[3],
    )
    encoder_stride = 32
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride, if_stu=True)
    model.num_init_heads = 3
    return model

@register_model
def simmim_swin_tiny_d13(pretrained=False, decoder_idx=None, **kwargs):
    encoder = SwinTransformerForSimMIM(
        img_size = 224,
        embed_dim = 48,
        window_size = 7,
        depths=[2, 2, 6, 3], 
        num_heads=[ 3, 6, 12, 24 ],
        num_classes=0,
        decoder_idx=decoder_idx,
    )
    encoder_stride = 32 // max(1, (2 ** (2-decoder_idx[0])))
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride, if_stu=True)
    model.num_init_heads = 3
    return model

@register_model
def simmim_swin_small(pretrained=False, **kwargs):
    encoder = SwinTransformerForSimMIM(
        img_size = 224,
        embed_dim = 96,
        window_size = 7,
        depths=[2, 2, 6, 2], 
        num_heads=[ 3, 6, 12, 24 ],
        num_classes=0,
        decoder_idx=[3],
    )
    encoder_stride = 32
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
    model.num_init_heads = 3
    return model

@register_model
def simmim_swin_base(pretrained=False, **kwargs):
    encoder = SwinTransformerForSimMIM(
        img_size=192,
        embed_dim = 128,
        window_size = 6,
        depths=[ 2, 2, 18, 2 ], 
        num_heads=[ 4, 8, 16, 32 ],
        num_classes=0,
        decoder_idx=[3],
    )
    encoder_stride = 32
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
    model.num_init_heads = 4
    return model
