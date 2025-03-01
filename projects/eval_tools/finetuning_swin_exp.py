# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
from finetuning_mae_exp import Exp as BaseExp
from loguru import logger
from util.lr_decay import create_optimizer, LayerDecayValueAssigner_swin
from swin_transformer import *
import os
from scipy import interpolate
import numpy as np
import torch


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
        self.weights_prefix = "model.encoder"
        # self.print_interval = 10
        # self.enable_tensorboard = True
        self.save_folder_prefix = "ft_"
        
        self.encoder_arch = "swin_tiny_224"

    def get_optimizer(self):
        if "optimizer" not in self.__dict__:
            if "lr" not in self.__dict__:
                self.lr = self.basic_lr_per_img * self.batch_size

            depths = self.get_model().model.depths
            num_layers = sum(depths)
            if self.layer_decay < 1.0:
                assigner = LayerDecayValueAssigner_swin(
                    list(self.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)), depths
                )
            else:
                assigner = None

            if assigner is not None:
                logger.info("Assigned values = %s" % str(assigner.values))

            skip_weight_decay_list = self.get_model().model.no_weight_decay()
            logger.info("Skip weight decay list: {}".format(skip_weight_decay_list))

            self.optimizer = create_optimizer(
                self,
                self.get_model().model,
                skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None,
                get_layer_scale=assigner.get_scale if assigner is not None else None,
            )
        return self.optimizer
    
    def set_model_weights(self, ckpt_path, map_location="cpu"):
        if not os.path.isfile(ckpt_path):
            from torch.nn.modules.module import _IncompatibleKeys

            logger.info("No checkpoints found! Training from scratch!")
            return _IncompatibleKeys(missing_keys=None, unexpected_keys=None)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        weights_prefix = self.weights_prefix
        if not weights_prefix:
            state_dict = {"model." + k: v for k, v in ckpt["model"].items()}
        else:
            if weights_prefix and not weights_prefix.endswith("."):
                weights_prefix += "."
            if all(key.startswith("module.") for key in ckpt["model"].keys()):
                weights_prefix = "module." + weights_prefix
            state_dict = {k.replace(weights_prefix, "model."): v for k, v in ckpt["model"].items() if 'tch_model' not in k}
            del state_dict['model.norm.weight']
            del state_dict['model.norm.bias']
            state_dict = self.remap_pretrained_keys_swin(self.get_model().state_dict(), state_dict)
        msg = self.get_model().load_state_dict(state_dict, strict=False)
        return msg

    def remap_pretrained_keys_swin(self, state_dict, checkpoint_model):
        # state_dict = model.state_dict()
        
        # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_bias_table" in key:
                relative_position_bias_table_pretrained = checkpoint_model[key]
                if key not in state_dict:
                    continue
                relative_position_bias_table_current = state_dict[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.info(f"Error in loading {key}, passing......")
                else:
                    if L1 != L2:
                        logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                        src_size = int(L1 ** 0.5)
                        dst_size = int(L2 ** 0.5)

                        def geometric_progression(a, r, n):
                            return a * (1.0 - r ** n) / (1.0 - r)

                        left, right = 1.01, 1.5
                        while right - left > 1e-6:
                            q = (left + right) / 2.0
                            gp = geometric_progression(1, q, src_size // 2)
                            if gp > dst_size // 2:
                                right = q
                            else:
                                left = q

                        # if q > 1.090307:
                        #     q = 1.090307

                        dis = []
                        cur = 1
                        for i in range(src_size // 2):
                            dis.append(cur)
                            cur += q ** (i + 1)

                        r_ids = [-_ for _ in reversed(dis)]

                        x = r_ids + [0] + dis
                        y = r_ids + [0] + dis

                        t = dst_size // 2.0
                        dx = np.arange(-t, t + 0.1, 1.0)
                        dy = np.arange(-t, t + 0.1, 1.0)

                        logger.info("Original positions = %s" % str(x))
                        logger.info("Target positions = %s" % str(dx))

                        all_rel_pos_bias = []

                        for i in range(nH1):
                            z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                            f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                            all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                                relative_position_bias_table_pretrained.device))

                        new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                        checkpoint_model[key] = new_rel_pos_bias

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del checkpoint_model[k]

        # delete relative_coords_table since we always re-init it
        relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
        for k in relative_coords_table_keys:
            del checkpoint_model[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del checkpoint_model[k]

        return checkpoint_model



if __name__ == "__main__":
    exp = Exp(2)
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
