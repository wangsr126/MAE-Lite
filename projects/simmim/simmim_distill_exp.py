# --------------------------------------------------------
# Copyright (c) 2025 Institute of Automation Chinese Academy of Sciences. All Rights Reserved.
# --------------------------------------------------------
import os, sys
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss

from timm.models import create_model

from simmim_exp import Exp as BaseExp, SimMIM
from loguru import logger
from projects.eval_tools.util.forward_hook_swin import AttnCatcher, RepCatcher
import copy
import numpy as np

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

class SimMIM_distill(SimMIM):
    def __init__(self, args, model, tch_model):
        super(SimMIM_distill, self).__init__(args, model)
        self.tch_model = tch_model
        for p in self.tch_model.parameters():
            p.requires_grad = False
        self.tch_model.eval()
        self.distill_criterion = MSELoss()
        self.mask_generator = MaskGenerator(
            input_size=224,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=args.mask_ratio,
        )
        # self.distill_criterion = nn.CrossEntropyLoss()
        # attn
        self.distill_attn_idx = args.distill_attn_idx
        if self.distill_attn_idx is not None and len(self.distill_attn_idx) > 0:
            self.attn_alpha = args.distill_attn_alpha
            self.attn_catcher = AttnCatcher(self.model.encoder, [(3,2)])
            # TODO: align teacher and student layers when the depths are different.
            self.tch_distill_attn_idx = args.tch_distill_attn_idx
            self.tch_attn_catcher = AttnCatcher(self.tch_model.encoder, [(3,1)])
            use_attn_adapter = args.use_attn_adapter
            if self.model.num_init_heads != self.tch_model.num_init_heads:
                use_attn_adapter = True
            if use_attn_adapter:
                self.attn_adapter = nn.ModuleDict()
                temp_scale = 8
                for idx in self.distill_attn_idx:
                    self.attn_adapter["adapter{}".format(idx-1)] = nn.Conv2d(self.model.num_init_heads*temp_scale, self.tch_model.num_init_heads*temp_scale, 1, bias=False)
                    temp_scale = temp_scale // 2
            else:
                self.attn_adapter = nn.ModuleDict({"adapter{}".format(idx-1): nn.Identity() for idx in self.distill_attn_idx})
        # hidden
        self.distill_hidden_idx = args.distill_hidden_idx
        if self.distill_hidden_idx is not None and len(self.distill_hidden_idx) > 0:
            self.hidden_alpha = args.distill_hidden_alpha
            self.hidden_catcher = RepCatcher(self.model, self.distill_hidden_idx)
            # TODO: align teacher and student layers when the depths are different.
            tch_distill_hidden_idx = args.tch_distill_hidden_idx
            self.tch_distill_hidden_idx = tch_distill_hidden_idx
            self.tch_hidden_catcher = RepCatcher(self.tch_model, tch_distill_hidden_idx)
            use_hidden_adapter = args.use_hidden_adapter
            if self.model.embed_dim != self.tch_model.embed_dim:
                use_hidden_adapter = True
            if use_hidden_adapter:
                self.hidden_adapter = nn.ModuleDict(
                    {
                        "adapter{}".format(idx): nn.Linear(self.model.blocks[idx-1].dim_out, self.tch_model.blocks[tch_distill_hidden_idx[iter]-1].dim_out, bias=False)
                        for iter,idx in enumerate(self.distill_hidden_idx)
                    }
                )
            else:
                self.hidden_adapter = nn.ModuleDict({"adapter{}".format(idx): nn.Identity() for idx in self.distill_hidden_idx})

    def train(self, mode):
        super().train(mode)
        self.tch_model.eval()

    def get_distill_attn_loss(self):
        loss_attn = 0
        if len(self.distill_attn_idx) > 0 and len(self.tch_distill_attn_idx) > 0:
            for i in range(len(self.distill_attn_idx)):
                idx = self.distill_attn_idx[i]
                tch_idx = self.tch_distill_attn_idx[i]
                adapter = self.attn_adapter["adapter{}".format(idx-1)]
                attn = self.attn_catcher.get_features()[0]
                tch_attn = self.tch_attn_catcher.get_features()[0].detach()
                loss_attn += self.distill_criterion(adapter(attn), tch_attn)
        return loss_attn
    
    def get_distill_hidden_loss(self):
        loss_hidden = 0
        if len(self.distill_hidden_idx) > 0:
            for i in range(len(self.distill_hidden_idx)):
                idx = self.distill_hidden_idx[i]
                tch_idx = self.tch_distill_hidden_idx[i]
                adapter = self.hidden_adapter["adapter{}".format(idx)]
                hidden = self.hidden_catcher.get_features(idx, remove_cls_token=False)
                tch_hidden = self.tch_hidden_catcher.get_features(tch_idx, remove_cls_token=False).detach()
                loss_hidden += self.distill_criterion(adapter(hidden), tch_hidden)
        return loss_hidden

    def forward(self, x, target=None, update_param=False):
        if self.training:
            mask_list = []
            for _ in range(len(x)):
                mask = torch.tensor(self.mask_generator(),dtype=torch.int32).cuda(non_blocking=True).unsqueeze(0)
                mask_list.append(mask)
            mask = torch.cat(mask_list,dim=0)
            images = x
            if self.mixup_fn is not None:
                images, _ = self.mixup_fn(images, target)
            loss = self.model(images, mask)
            
            with torch.no_grad():
                _ = self.tch_model.forward_encoder(
                    images, mask
                )

            output_dict = {}
            loss_attn = self.get_distill_attn_loss()
            if isinstance(loss_attn, torch.Tensor):
                output_dict["attn"] = loss_attn.detach().item()
                loss += loss_attn * self.attn_alpha
            loss_hidden = self.get_distill_hidden_loss()
            if isinstance(loss_hidden, torch.Tensor):
                output_dict['hidden'] = loss_hidden.detach().item()
                loss += loss_hidden * self.hidden_alpha
            
            if self.ema_model is not None:
                self.ema_model.update(self.model)
            return loss, output_dict
        else:
            raise NotImplementedError


def set_model_weights(model, ckpt_path, weights_prefix):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # key = "model"
    key = "state_dict"
    if not weights_prefix:
        state_dict = ckpt[key]
    else:
        if weights_prefix and not weights_prefix.endswith("."):
            weights_prefix += "."
        if all(key.startswith("module.") for key in ckpt[key].keys()):
            weights_prefix = "module." + weights_prefix
        state_dict = {k.replace(weights_prefix, ""): v for k, v in ckpt[key].items()}

    msg = model.encoder.load_state_dict(state_dict, strict=False)
    del state_dict
    return msg

class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=400):
        super(Exp, self).__init__(batch_size, max_epoch)
        self.encoder_arch = "simmim_swin_tiny"
        self.mask_ratio = 0.6
        self.distill_attn_alpha = 1.0
        self.distill_attn_idx = (13,)  # 1-12
        self.tch_distill_attn_idx = (12,)  # 1-12
        self.use_attn_adapter = True
        self.distill_hidden_alpha = 1.0
        self.distill_hidden_idx = ()  # 1-12
        self.tch_distill_hidden_idx = ()  # 1-12
        self.use_hidden_adapter = True

        self.teacher_arch = "simmim_swin_small"
        self.encoder_arch = "simmim_swin_tiny_d13"
        self.stu_decoder_idx = [1] # 0-3
        self.teacher_weights_prefix = ""
        self.teacher_ckpt_path = "~/projects/mae-lite/checkpoints/mae_base_1600e.pth.tar"
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]

    def get_model(self):
        if "model" not in self.__dict__:
            model = create_model(self.encoder_arch, decoder_idx=self.stu_decoder_idx)
            tch_model = create_model(self.teacher_arch)
            del tch_model.decoder
            msg = set_model_weights(tch_model, self.teacher_ckpt_path, self.teacher_weights_prefix)
            logger.info("Model params {} are not loaded".format(msg.missing_keys))
            logger.info("State-dict params {} are not used".format(msg.unexpected_keys))
            self.model = SimMIM_distill(self, model, tch_model)
        return self.model


if __name__ == "__main__":
    exp = Exp(2, 1)
    exp.num_workers = 1
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
    train_loader = loader["train"]
    for inps in train_loader:
        images, target = inps
        out = model(images, target=target)
