# --------------------------------------------------------
# Copyright (c) 2025 Institute of Automation Chinese Academy of Sciences. All Rights Reserved.
# --------------------------------------------------------
import os
import torch
from torch import nn

from timm.data import Mixup
from timm.models import create_model
from timm.utils import ModelEmaV2

from mae_lite.exps.timm_imagenet_exp import Exp as BaseExp
from mae_lite.layers import build_lr_scheduler
from simmim import *
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

class SimMIM(nn.Module):
    def __init__(self, args, model):
        super(SimMIM, self).__init__()
        self.model = model
        # mixup
        mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.num_classes,
            )
        else:
            mixup_fn = None
        self.mixup_fn = mixup_fn
        self.mask_ratio = args.mask_ratio
        # ema
        if args.model_ema:
            self.ema_model = ModelEmaV2(
                self.model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else None
            )
        else:
            self.ema_model = None
        
        self.mask_generator = MaskGenerator(
            input_size=224,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=args.mask_ratio,
        )

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

            if self.ema_model is not None:
                self.ema_model.update(self.model)
            return loss, None
        else:
            raise NotImplementedError


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=400):
        super(Exp, self).__init__(batch_size, max_epoch)
        # MAE
        self.norm_pix_loss = True
        self.mask_ratio = 0.6
        # self.input_size = (3, 192, 192)

        # dataset & model
        self.dataset = "ImageNet"
        self.encoder_arch = "simmim_swin_tiny"

        # optimizer
        self.opt = "adamw"
        self.opt_eps = 1e-8
        self.opt_betas = (0.9, 0.99)
        self.momentum = 0.9
        self.weight_decay = 0.05
        self.clip_grad = 5.0
        self.clip_mode = "norm"

        # schedule
        self.sched = "warmmultistep"
        self.basic_lr_per_img = 4.0e-4 / 2048
        self.warmup_lr_per_img = 2.0e-6 /2048
        self.min_lr = 0.0
        self.warmup_epochs = 10
        self.multi_steps = [700,]

        # augmentation & regularization
        self.no_aug = False
        self.scale = (0.67, 1.0)
        self.ratio = (3.0 / 4, 4.0 / 3.0)
        self.hflip = 0.5
        self.vflip = 0.0
        self.color_jitter = None
        self.smoothing = 0.0

        # self.num_workers = 10
        self.weights_prefix = "model"
        # self.print_interval = 10
        # self.enable_tensorboard = True
        self.save_folder_prefix = ""
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]

    def get_model(self):
        if "model" not in self.__dict__:
            model = create_model(self.encoder_arch, norm_pix_loss=self.norm_pix_loss)
            if self.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = SimMIM(self, model)
        return self.model

    def get_data_loader(self):
        super().get_data_loader()
        self.data_loader["eval"] = None
        return self.data_loader

    def get_lr_scheduler(self):
        if "lr" not in self.__dict__:
            self.lr = self.basic_lr_per_img * self.batch_size
        if "warmup_lr" not in self.__dict__:
            self.warmup_lr = self.warmup_lr_per_img * self.batch_size
        if "min_lr" not in self.__dict__:
            self.min_lr = self.min_lr_per_img * self.batch_size

        optimizer = self.get_optimizer()
        iters_per_epoch = len(self.get_data_loader()["train"])
        scheduler = build_lr_scheduler(
            self.sched,
            optimizer,
            self.lr,
            total_steps=iters_per_epoch * self.max_epoch,
            milestones=[iters_per_epoch * k for k in self.multi_steps],
            warmup_steps=iters_per_epoch * self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
        )
        return scheduler


if __name__ == "__main__":
    exp = Exp(2)
    print(exp.exp_name)
    loader = exp.get_data_loader()
    model = exp.get_model()
    print(model)
    opt = exp.get_optimizer()
    sched = exp.get_lr_scheduler()
