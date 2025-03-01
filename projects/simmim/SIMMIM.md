## Experiments on [SimMIM](https://arxiv.org/abs/2111.09886)

### Preparation
Download the teacher [Swin-Small](https://drive.google.com/file/d/152Z_pjzwguIxpKs2TEzRj7Lau0T7RKys/view?usp=drive_link) to `<BASE_FOLDER>/checkpoints/swin_small_800e.pth.tar`

### Pre-Training
```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/simmim
ssl_train -b 2048 -d 0-7 -e 400 -f ./simmim_exp.py --amp \
--exp-options exp_name=simmim/simmim_swin_tiny_400e
```

### Pre-Training with Decoupled Distillation
```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/simmim
ssl_train -b 2048 -d 0-7 -e 400 -f ./simmim_distill_exp.py --amp --exp-options \
teacher_ckpt_path="<BASE_FOLDER>/checkpoints/swin_small_800e.pth.tar" exp_name=simmim/simmim_swin_tiny_d2_distill_400e
```

### Fine-Tuning
```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -f finetuning_swin_exp.py --amp [--ckpt <checkpoint-path>] \
--exp-options pretrain_exp_name=simmim/simmim_swin_tiny_d2_distill_400e
```

# Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| [simmim](simmim_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1biTTlfMau8XFmmysIaqwcqDER8iKDP-K/view?usp=sharing) |
| [simmim](simmim_exp.py) | 400 | [impr.](../eval_tools/finetuning_swin_exp.py) | 300 | 77.4 | [link](https://drive.google.com/file/d/1zs6pnnoN5hZcBSZrSKlwVXJhHpNI312g/view?usp=sharing) |
| [simmim_d2_distill](simmim_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1ocdF-sXwhiWjPLx3uTaej65arPJApFYA/view?usp=sharing) |
| [simmim_d2_distill](simmim_distill_exp.py) | 400 | [impr.](../eval_tools/finetuning_swin_exp.py) | 300 | 77.8 | [link](https://drive.google.com/file/d/1ywSSfR3o7vfuPpk-MfAJDSzvw-0HI5Fl/view?usp=sharing) |
