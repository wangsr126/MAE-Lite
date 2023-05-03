# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from timm.models import create_model
from finetuning_mae_exp import Exp as BaseExp
from mae_lite.exps.timm_imagenet_exp import Model as BaseModel


class ModelDistill(BaseModel):
    def __init__(self, args, model, tch_model=None):
        super(ModelDistill, self).__init__(args, model)
        self.tch_model = tch_model
        if tch_model is not None:
            for p in self.tch_model.parameters():
                p.requires_grad = False
            self.tch_model.eval()
        self.distill_type = args.distill_type  # none, soft, hard
        assert self.distill_type in ["none", "soft", "hard"]
        if self.distill_type in ["soft", "hard"]:
            assert self.tch_model is not None
        self.alpha = args.distill_alpha
        self.tau = args.distill_tau

    def train(self, mode):
        super().train(mode)
        if self.tch_model is not None:
            self.tch_model.eval()

    def forward(self, x, target=None, update_param=False):
        if self.training:
            if self.mixup_fn is not None:
                x, target = self.mixup_fn(x, target)
            outputs = self.model(x)
            outputs_kd = None
            if not isinstance(outputs, torch.Tensor):
                outputs, outputs_kd = outputs
            loss = self.train_loss_fn(outputs, target)

            if outputs_kd is not None:
                with torch.no_grad():
                    tch_outputs = self.tch_model(x)
                if self.distill_type == "soft":
                    T = self.tau
                    distill_loss = (
                        F.kl_div(
                            F.log_softmax(outputs_kd / T, dim=1),
                            F.log_softmax(tch_outputs / T, dim=1),
                            reduction="sum",
                            log_target=True,
                        )
                        * (T * T)
                        / outputs_kd.numel()
                    )
                elif self.distill_type == "hard":
                    distill_loss = F.cross_entropy(outputs_kd, tch_outputs.argmax(dim=1))
                else:
                    raise KeyError
                loss = loss * (1 - self.alpha) + distill_loss * self.alpha

            if self.ema_model is not None:
                self.ema_model.update(self.model)
            return loss, None
        else:
            logits = self.model(x)
            return logits


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)

        # distill
        self.distilled = True
        self.teacher_arch = "regnety_160"
        self.distill_type = "hard"
        self.distill_alpha = 0.5
        self.distill_tau = 1.0

        self.weights_prefix = "model"
        # self.print_interval = 10
        # self.enable_tensorboard = True
        self.save_folder_prefix = "ft_distill_"
        # self.eval_interval = 1

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
                distilled=self.distilled,
            )
            tch_model = (
                create_model(
                    self.teacher_arch,
                    pretrained=True,
                    num_classes=self.num_classes,
                    global_pool="avg",
                )
                if self.teacher_arch is not None
                else None
            )
            self.model = ModelDistill(self, encoder, tch_model)
        return self.model


if __name__ == "__main__":
    exp = Exp(2, 1)
    exp.mixup = 0.0
    exp.cutmix = 0.0
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
    inp = torch.randn(2, 3, 224, 224)
    target = torch.zeros(2, dtype=torch.int64)
    out = model(inp, target)
