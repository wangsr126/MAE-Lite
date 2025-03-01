# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
from finetuning_exp import Exp as BaseExp
from timm.models import create_model
from mae_lite.exps.timm_imagenet_exp import  Model


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)
        self.encoder_arch = "vit_tiny_patch16_rpe"
        self.use_abs_pos_emb = True
        self.use_rel_pos_bias = True
        self.use_shared_rel_pos_bias = False
        self.qkv_bias = True
        self.qv_bias = True
        self.save_folder_prefix = "ft_impr_rpe_"

    def get_model(self):
        if "model" not in self.__dict__:
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
                qkv_bias=self.qkv_bias,
                qv_bias=self.qv_bias,
            )
            self.model = Model(self, encoder)
        return self.model


if __name__ == "__main__":
    exp = Exp(2)
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
