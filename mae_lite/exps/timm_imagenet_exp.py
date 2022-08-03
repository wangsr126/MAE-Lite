# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
""" 
Support timm training (timm==0.4.12)
Differences:
* lr, warmup_lr, min_lr are replaced by basic_lr_per_img, warmup_lr_per_img, min_lr_per_img
* Some features are not supported now, eg., aug_repeats, jsd_loss, bce_loss, split_bn
"""
import os
import torch
from torch import nn

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm import create_model
from timm.data import create_loader, resolve_data_config, Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler as create_scheduler_timm
from timm.optim import create_optimizer
from timm.utils import ModelEmaV2

from mae_lite.exps.base_exp import BaseExp
from mae_lite.data import build_dataset
from mae_lite.data.transforms import transforms, ToRGB
from mae_lite.layers.lr_scheduler import LRScheduler, _Scheduler
import mae_lite.utils.torch_dist as dist


class Model(nn.Module):
    def __init__(self, args, model):
        super(Model, self).__init__()
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

        # criterion
        if mixup_active:
            train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        self.train_loss_fn = train_loss_fn
        # ema
        if args.model_ema:
            self.ema_model = ModelEmaV2(
                self.model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else None
            )
            for p in self.ema_model.parameters():
                p.requires_grad = False
        else:
            self.ema_model = None

    def forward(self, x, target=None, update_param=False):
        if self.training:
            if self.mixup_fn is not None:
                x, target = self.mixup_fn(x, target)
            logits = self.model(x)
            loss = self.train_loss_fn(logits, target)
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            # TODO: accuracy monitor
            # top1, top5 = accuracy(logits, target, (1, 5))
            return loss, None
        else:
            logits = self.model(x)
            return logits


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)
        # dataset & model
        self.dataset = "ImageNet"
        self.encoder_arch = "resnet50"
        self.pretrained = False
        self.num_classes = 1000
        self.global_pool = None
        self.img_size = None
        self.input_size = None
        self.crop_pct = None
        self.mean = None
        self.std = None
        self.interpolation = ""
        self.validation_batch_size = None
        self.validation_dataset = None

        # optimizer
        self.opt = "sgd"
        self.opt_eps = None
        self.opt_betas = None
        self.momentum = 0.9
        self.weight_decay = 2e-5
        self.clip_grad = None
        self.clip_mode = "norm"

        # schedule
        self.sched = "cosine"
        # self.lr = 0.05
        self.basic_lr_per_img = 0.05 / 128
        self.lr_noise = None
        self.lr_noise_pct = 0.67
        self.lr_noise_std = 1.0
        self.lr_cycle_mul = 1.0
        self.lr_cycle_decay = 0.5
        self.lr_cycle_limit = 1
        self.lr_k_decay = 1.0
        # self.warmup_lr = 0.0001
        self.warmup_lr_per_img = 0.0001 / 128
        # self.min_lr = 1e-6
        self.min_lr_per_img = 1e-6 / 128
        self.epochs = max_epoch
        self.epoch_repeats = 0
        self.start_epoch = None  #
        self.decay_epochs = None
        self.warmup_epochs = 3
        self.cooldown_epochs = 10
        self.patience_epochs = 10
        self.decay_rate = 0.1

        # augmentation & regularization
        self.no_aug = False
        self.scale = (0.08, 1.0)
        self.ratio = (3.0 / 4, 4.0 / 3.0)
        self.hflip = 0.5
        self.vflip = 0.0
        self.color_jitter = 0.4
        self.aa = None
        # self.aug_repeats = 0  # not support
        # self.aug_splits = 0  # not support
        # self.jsd_loss = False  # not support
        # self.bce_loss = False  # not support
        self.reprob = 0.0
        self.remode = "pixel"
        self.recount = 1
        self.resplit = False
        self.mixup = 0.0
        self.cutmix = 0.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = "batch"
        self.mixup_off_epoch = 0
        self.smoothing = 0.1
        self.train_interpolation = "random"
        self.drop = 0.0
        self.drop_connect = None
        self.drop_path = None
        self.drop_block = None

        # batch norm
        self.bn_tf = False
        self.bn_momentum = None
        self.bn_eps = None
        self.sync_bn = False
        self.dist_bn = "reduce"
        # self.split_bn = False  # not support

        # EMA
        self.model_ema = False
        self.model_ema_force_cpu = False
        self.model_ema_decay = 0.9998

        self.seed = 0
        self.num_workers = 10
        self.weights_prefix = "model"
        self.print_interval = 10
        self.enable_tensorboard = True
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]

    def get_model(self):
        if "model" not in self.__dict__:
            model = create_model(
                self.encoder_arch,
                pretrained=self.pretrained,
                num_classes=self.num_classes,
                drop_rate=self.drop,
                drop_connect_rate=self.drop_connect,
                drop_path_rate=self.drop_path,
                drop_block_rate=self.drop_block,
                global_pool=self.global_pool,
                bn_tf=self.bn_tf,
                bn_momentum=self.bn_momentum,
                bn_eps=self.bn_eps,
            )
            if self.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = Model(self, model)
        return self.model

    def get_data_loader(self):
        if "data_loader" not in self.__dict__:
            dataset_train = build_dataset(self.dataset, True)
            dataset_eval = build_dataset(self.validation_dataset if self.validation_dataset else self.dataset, False)

            batch_size_per_gpu = self.batch_size // dist.get_world_size()
            data_config = resolve_data_config(vars(self), model=self.get_model().model, verbose=dist.is_main_process())
            loader_train = create_loader(
                dataset_train,
                input_size=data_config["input_size"],
                batch_size=batch_size_per_gpu,
                is_training=True,
                use_prefetcher=False,
                no_aug=self.no_aug,
                re_prob=self.reprob,
                re_mode=self.remode,
                re_count=self.recount,
                re_split=self.resplit,
                scale=self.scale,
                ratio=self.ratio,
                hflip=self.hflip,
                vflip=self.vflip,
                color_jitter=self.color_jitter,
                auto_augment=self.aa,
                interpolation=self.train_interpolation,
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=self.num_workers,
                distributed=dist.is_distributed(),
                pin_memory=False,
            )
            validation_batch_size_per_gpu = (self.validation_batch_size or self.batch_size) // dist.get_world_size()
            loader_eval = create_loader(
                dataset_eval,
                input_size=data_config["input_size"],
                batch_size=validation_batch_size_per_gpu,
                is_training=False,
                use_prefetcher=False,
                interpolation=data_config["interpolation"],
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=self.num_workers,
                distributed=dist.is_distributed(),
                crop_pct=data_config["crop_pct"],
                pin_memory=False,
            )
            loader_train.dataset.transform = transforms.Compose([ToRGB(), loader_train.dataset.transform])
            loader_eval.dataset.transform = transforms.Compose([ToRGB(), loader_eval.dataset.transform])
            self.data_loader = {"train": loader_train, "eval": loader_eval}
        return self.data_loader

    def get_optimizer(self):
        if "optimizer" not in self.__dict__:
            if "lr" not in self.__dict__:
                self.lr = self.basic_lr_per_img * self.batch_size
            self.optimizer = create_optimizer(self, self.get_model())
        return self.optimizer

    def get_lr_scheduler(self):
        if "lr" not in self.__dict__:
            self.lr = self.basic_lr_per_img * self.batch_size
        if "warmup_lr" not in self.__dict__:
            self.warmup_lr = self.warmup_lr_per_img * self.batch_size
        if "min_lr" not in self.__dict__:
            self.min_lr = self.min_lr_per_img * self.batch_size
        if "epochs" not in self.__dict__:
            self.epochs = self.max_epoch
        optimizer = self.get_optimizer()
        iters_per_epoch = len(self.get_data_loader()["train"])
        scheduler = TimmLRScheduler(self, optimizer, interval=iters_per_epoch)
        return scheduler


class TimmLRScheduler(LRScheduler):
    def __init__(self, args, optimizer, interval=1):
        self.scheduler_timm, _ = create_scheduler_timm(args, optimizer)
        super(TimmLRScheduler, self).__init__(optimizer, _Scheduler(0.0, 1), interval)

    def step(self, count):
        count, inner_count = divmod(count, self.interval)
        if inner_count == 0:
            self.scheduler_timm.step(count)

    def state_dict(self):
        return self.scheduler_timm.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler_timm.load_state_dict(state_dict)
