# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
from finetuning_mae_exp import Exp as BaseExp
from hiera import *


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)

        # optimizer
        self.clip_grad = None
        # self.clip_mode = "norm"

        # augmentation & regularization
        self.color_jitter = 0.3
        self.aa = "rand-m10-mstd0.5-inc1"
        self.reprob = 0.0
        self.mixup = 0.2
        self.cutmix = 0.0
        self.smoothing = 0.0
        self.drop_path = 0.0

        # self.num_workers = 10
        self.weights_prefix = "model"
        # self.print_interval = 10
        # self.enable_tensorboard = True
        self.save_folder_prefix = "ft_"


if __name__ == "__main__":
    exp = Exp(2)
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
