# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os
import torch
from timm.models import create_model
from finetuning_mae_exp import Exp as BaseExp, Model
from loguru import logger


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)
        self.layer_decay = 0.75
        self.global_pool = True
        self.dataset = "CIFAR100"
        self.num_classes = 100
        self.opt = "sgd"
        self.opt_eps = None
        self.opt_betas = None
        self.clip_grad = 1.0
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.basic_lr_per_img = 0.03 / 512
        self.warmup_epochs = 20
        # augmentation & regularization
        # self.no_aug = False
        # self.scale = (0.08, 1.0)
        # self.ratio = (3./4, 4./3.)
        # self.hflip = 0.5
        # self.vflip = 0.
        self.color_jitter = None
        self.aa = None
        self.reprob = 0.0
        self.mixup = 0.0
        self.cutmix = 0.0
        self.smoothing = 0.0
        self.train_interpolation = "bicubic"
        # self.drop = 0.0
        # self.drop_connect = None
        self.drop_path = 0.0
        self.attn_drop_rate = 0.0
        # self.drop_block = None
        self.encoder_arch = "vit_tiny_patch16"
        self.save_folder_prefix = "ft_cifar100_"
        self.eval_interval = 10

        self.use_abs_pos_emb = True
        self.use_rel_pos_bias = False
        self.use_shared_rel_pos_bias = False
        self.qkv_bias = True
        self.qv_bias = True
        self.init_values = None

    def get_model(self):
        if "model" not in self.__dict__:
            if self.encoder_arch == 'vit_tiny_patch16_rpe':
                encoder = create_model(
                    self.encoder_arch,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                    drop_rate=self.drop,
                    drop_path_rate=self.drop_path,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_block_rate=self.drop_block,
                    global_pool=self.global_pool,
                    use_abs_pos_emb=self.use_abs_pos_emb,
                    use_rel_pos_bias=self.use_rel_pos_bias,
                    use_shared_rel_pos_bias=self.use_shared_rel_pos_bias,
                    init_values=self.init_values,
                    qkv_bias=self.qkv_bias,
                    qv_bias=self.qv_bias,
                )
            else:
                encoder = create_model(
                    self.encoder_arch,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                    drop_rate=self.drop,
                    drop_path_rate=self.drop_path,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_block_rate=self.drop_block,
                    global_pool=self.global_pool,
                )
            self.model = Model(self, encoder)
        return self.model

    def set_model_weights(self, ckpt_path, map_location="cpu"):
        BLACK_LIST = ("head", )

        def _match(key):
            return any([k in key for k in BLACK_LIST])

        if not os.path.isfile(ckpt_path):
            from torch.nn.modules.module import _IncompatibleKeys

            logger.info("No checkpoints found! Training from scratch!")
            return _IncompatibleKeys(missing_keys=None, unexpected_keys=None)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        weights_prefix = self.weights_prefix
        if not weights_prefix:
            state_dict = {"model." + k: v for k, v in ckpt["model"].items() if not _match(k)}
        else:
            if weights_prefix and not weights_prefix.endswith("."):
                weights_prefix += "."
            if all(key.startswith("module.") for key in ckpt["model"].keys()):
                weights_prefix = "module." + weights_prefix
            state_dict = {k.replace(weights_prefix, "model."): v for k, v in ckpt["model"].items() if not _match(k)}
        msg = self.get_model().load_state_dict(state_dict, strict=False)
        return msg


if __name__ == "__main__":
    exp = Exp(2, 1)
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
