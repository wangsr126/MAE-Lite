## Experiments on [Hiera](https://arxiv.org/abs/2306.00989/)

### Preparation
Download the teacher [Hiera-Base](https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth) to `<BASE_FOLDER>/checkpoints/mae_hiera_base_224.pth`

### Pre-Training
```bash
# 4096 batch-sizes on 8 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -d 0-7 -e 400 -f ./hiera_mae_exp.py --amp \
--exp-options exp_name=hiera/mae_hiera_tiny_400e
```

### Pre-Training with Distillation
```bash
# 4096 batch-sizes on 8 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -d 0-7 -e 400 -f ./hiera_mae_distill_exp.py --amp --exp-options \
teacher_ckpt_path="<BASE_FOLDER>/checkpoints/mae_hiera_base_224.pth" exp_name=hiera/mae_hiera_tiny_distill_400e
```

### Fine-Tuning
```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -f finetuning_exp.py --amp [--ckpt <checkpoint-path>] \
--exp-options pretrain_exp_name=hiera/mae_hiera_tiny_distill_400e encoder_arch="hiera_tiny_224"
```

# Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| [hiera_mae](hiera_mae_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1dXL6lc40kuySP3aSnz0kbddmpebTENaz/view?usp=sharing) |
| [hiera_mae](hiera_mae_exp.py) | 400 | [impr.](../eval_tools/finetuning_hiera_exp.py) | 300 | 78.5 | [link](https://drive.google.com/file/d/1s6Q6AZ0kV1YPVFjmxO12sgDr6PSJQpwq/view?usp=sharing) |
| [hiera_mae_distill](hiera_mae_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1s6Q6AZ0kV1YPVFjmxO12sgDr6PSJQpwq/view?usp=sharing) |
| [hiera_mae_distill](hiera_mae_distill_exp.py) | 400 | [impr.](../eval_tools/finetuning_hiera_exp.py) | 300 | 78.9 | [link](https://drive.google.com/file/d/1qknEZwHQoEfzGGOtildm6J8-IPoN4fgq/view?usp=sharing) |
