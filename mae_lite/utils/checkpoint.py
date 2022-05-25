# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os
import shutil
import torch


def save_checkpoint(state, is_best, save, model_name=""):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, model_name + "_best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)
