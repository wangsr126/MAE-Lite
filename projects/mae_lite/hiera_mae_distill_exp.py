# --------------------------------------------------------
# Copyright (c) 2025 Institute of Automation Chinese Academy of Sciences. All Rights Reserved.
# --------------------------------------------------------
import os, sys
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss

from timm.models import create_model

from hiera_mae_exp import Exp as BaseExp, Hiera
from loguru import logger
from projects.eval_tools.util.forward_hook import AttnCatcher, RepCatcher


class MAE_distill(Hiera):
    def __init__(self, args, model, tch_model):
        super(MAE_distill, self).__init__(args, model)
        self.tch_model = tch_model
        for p in self.tch_model.parameters():
            p.requires_grad = False
        self.tch_model.eval()
        self.distill_criterion = MSELoss()
        # self.distill_criterion = nn.CrossEntropyLoss()
        # attn
        self.distill_attn_idx = args.distill_attn_idx
        if self.distill_attn_idx is not None and len(self.distill_attn_idx) > 0:
            self.attn_alpha = args.distill_attn_alpha
            self.attn_catcher = AttnCatcher(self.model, self.distill_attn_idx)
            # TODO: align teacher and student layers when the depths are different.
            self.tch_distill_attn_idx = args.tch_distill_attn_idx
            self.tch_attn_catcher = AttnCatcher(self.tch_model, self.tch_distill_attn_idx)
            use_attn_adapter = args.use_attn_adapter
            if self.model.num_heads != self.tch_model.num_heads:
                use_attn_adapter = True
            if use_attn_adapter:
                self.attn_adapter = nn.ModuleDict()
                temp_scale = 8
                for idx in self.distill_attn_idx:
                    self.attn_adapter["adapter{}".format(idx-1)] = nn.Conv2d(self.model.num_heads*temp_scale, self.tch_model.num_heads*temp_scale, 1, bias=False)
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
                attn = self.attn_catcher.get_features(idx)
                tch_attn = self.tch_attn_catcher.get_features(tch_idx).detach()
                loss_attn += self.distill_criterion(adapter(attn.squeeze(2)), tch_attn.squeeze(2))
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
            images = x
            if self.mixup_fn is not None:
                images, _ = self.mixup_fn(images, target)
            loss, _, _, mask = self.model(images, self.mask_ratio, None)
            
            with torch.no_grad():
                _, _, = self.tch_model.forward_encoder(
                    images, self.mask_ratio, mask
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
    if not weights_prefix:
        state_dict = ckpt["model_state"]
    else:
        if weights_prefix and not weights_prefix.endswith("."):
            weights_prefix += "."
        if all(key.startswith("module.") for key in ckpt["model_state"].keys()):
            weights_prefix = "module." + weights_prefix
        state_dict = {k.replace(weights_prefix, ""): v for k, v in ckpt["model_state"].items()}

    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict
    return msg

class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=400):
        super(Exp, self).__init__(batch_size, max_epoch)
        self.encoder_arch = "mae_hiera_tiny_224"
        self.mask_ratio = 0.4
        self.distill_attn_alpha = 1.0
        self.distill_attn_idx = (12,)  # 1-13
        self.tch_distill_attn_idx = (24,)  # 1-24
        self.use_attn_adapter = True
        self.distill_hidden_alpha = 1.0
        self.distill_hidden_idx = ()  # 1-13
        self.tch_distill_hidden_idx = ()  # 1-24
        self.use_hidden_adapter = True

        self.teacher_arch = "mae_hiera_base_224"
        self.teacher_weights_prefix = ""
        self.teacher_ckpt_path = "~/projects/mae-lite/checkpoints/mae_hiera_base_224.pth.tar"
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]

    def get_model(self):
        if "model" not in self.__dict__:
            model = create_model(self.encoder_arch)
            tch_model = create_model(self.teacher_arch)
            del tch_model.mask_token
            del tch_model.decoder_embed
            del tch_model.decoder_pos_embed
            del tch_model.decoder_blocks
            del tch_model.decoder_norm
            del tch_model.decoder_pred
            msg = set_model_weights(tch_model, self.teacher_ckpt_path, self.teacher_weights_prefix)
            logger.info("Model params {} are not loaded".format(msg.missing_keys))
            logger.info("State-dict params {} are not used".format(msg.unexpected_keys))
            self.model = MAE_distill(self, model, tch_model)
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
