# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os
from functools import partial
import torch
import mae_lite.utils.torch_dist as dist

from mae_lite.data.sampler import InfiniteSampler
from mae_lite.exps.base_exp import BaseExp
from mae_lite.layers import build_lr_scheduler, build_optimizer
from mae_lite.data import build_dataset, build_transform
from mocov3 import MoCo_ViT
import vits


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=400):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ------------------------------------- model config ------------------------------ #
        self.moco_dim = 256
        self.moco_mlp_dim = 4096
        self.moco_m = 0.99
        self.moco_t = 0.2
        self.stop_grad_conv1 = False
        self.opt = "AdamW"

        # ------------------------------------ data loader config ------------------------- #
        self.image_size = 224
        self.data_num_workers = 6
        self.data_format = "list"

        # ------------------------------------  training config --------------------------- #
        self.basic_lr_per_img = 1.5e-4 / 256.0
        self.scheduler = "warmcos"
        self.warmup_epochs = 40
        self.warmup_lr = 1e-6

        self.weight_decay = 0.1
        self.momentum = 0.9
        self.print_interval = 10
        self.seed = 0
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]
        self.encoder_arch = "vit_tiny"
        self.dataset = "SSL_ImageNet"
        self.transform = "ssl_transform"
        self.enable_tensorboard = True

    def get_model(self):
        if "model" not in self.__dict__:
            model = MoCo_ViT(
                partial(vits.__dict__[self.encoder_arch], stop_grad_conv1=self.stop_grad_conv1),
                len(self.data_loader["train"]) * self.max_epoch,
                self.moco_dim,
                self.moco_mlp_dim,
                self.moco_t,
                self.moco_m,
            )
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = model
        return self.model

    def get_data_loader(self):
        if "data_loader" not in self.__dict__:

            transform = build_transform(self.transform, self.image_size)
            train_set = build_dataset(self.dataset, transform=transform)
            sampler = InfiniteSampler(len(train_set), shuffle=True, seed=self.seed if self.seed else 0)

            batch_size_per_gpu = self.batch_size // dist.get_world_size()

            dataloader_kwargs = {
                "num_workers": self.data_num_workers,
                "pin_memory": False,
            }
            dataloader_kwargs["sampler"] = sampler
            dataloader_kwargs["batch_size"] = batch_size_per_gpu
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["drop_last"] = True
            train_loader = torch.utils.data.DataLoader(train_set, **dataloader_kwargs)
            self.data_loader = {"train": train_loader, "eval": None}

        return self.data_loader

    def get_optimizer(self):
        # Noticing hear we only optimize student_encoder
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * self.batch_size

            self.optimizer = build_optimizer(
                self.opt,
                self.model.parameters(),
                lr=lr,
                weight_decay=self.weight_decay,
                # momentum=self.momentum,
            )
        return self.optimizer

    def get_lr_scheduler(self):
        if "lr" not in self.__dict__:
            self.lr = self.basic_lr_per_img * self.batch_size
        optimizer = self.get_optimizer()
        iters_per_epoch = len(self.get_data_loader()["train"])
        scheduler = build_lr_scheduler(
            self.scheduler,
            optimizer,
            self.lr,
            total_steps=iters_per_epoch * self.max_epoch,
            warmup_steps=iters_per_epoch * self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
        )
        return scheduler

    def set_current_state(self, current_step, **kwargs):
        self.get_model().set_current_state(current_step)
