## Experiments on ViT-Small

### Pre-Training
```bash
# 4096 batch-sizes on 8 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -d 0-7 -e 400 -f mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_small_400e encoder_arch=mae_vit_small_patch16
```

### Fine-Tuning on ImageNet
Please download the pre-trained models, *e.g.*, 

download [MAE-Small](https://drive.google.com/file/d/16polFU68dOe4YrmpgL9I1ekStzFG8XV5/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_small_400e.pth.tar`

Then, fine-tune the pre-trained model:

```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -d 0-7 -e 300 -f finetuning_exp.py --amp \
[--ckpt <checkpoint-path>] --exp-options pretrain_exp_name=mae_lite/mae_small_400e \
layer_decay=0.75 reprob=0.0 smoothing=0.0 encoder_arch=vit_small_patch16 color_jitter=0.3
```
- `<checkpoint-path>`: if set to `<BASE_FOLDER>/checkpoints/mae_small_400e.pth.tar`, it will be loaded as initialization; If not set, the checkpoint at `<BASE_FOLDER>/outputs/mae_lite/mae_small_400e/last_epoch_ckpt.pth.tar` will be loaded automatically.

### Pre-training with Distillation
**Preparation**

Download the teacher [MAE-Base](https://drive.google.com/file/d/1SPTjHIvw-yTOmw2ll-9cCiVyqR8NdPrX/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_base_1600e.pth.tar`

**Pre-Training**
```bash
# 4096 batch-sizes on 8 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -f mae_lite_distill_exp.py --amp --exp-options \
teacher_ckpt_path="<BASE_FOLDER>/checkpoints/mae_base_1600e.pth.tar" exp_name=mae_lite/mae_small_distill_400e encoder_arch=mae_vit_small_patch16
```

**Fine-Tuning**

```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -d 0-7 -e 300 -f finetuning_exp.py --amp \
[--ckpt <checkpoint-path>] --exp-options pretrain_exp_name=mae_lite/mae_small_distill_400e \
layer_decay=0.75 reprob=0.0 smoothing=0.0 encoder_arch=vit_small_patch16 color_jitter=0.3
```

## Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| - | - | [impr.](../eval_tools/finetuning_exp.py) | 300 | 80.2 | [link](https://drive.google.com/file/d/1VhiwQFfnB4WvRXH3GJcqEbM0GzVEikn3/view?usp=sharing) |
| [mae_lite](mae_lite_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/16polFU68dOe4YrmpgL9I1ekStzFG8XV5/view?usp=sharing) |
|  |  | [impr.](../eval_tools/finetuning_exp.py) | 300 | 82.1 | [link](https://drive.google.com/file/d/10UTc8eaPpudJhIDxKc23U97rEngHvH3J/view?usp=sharing) |
| [mae_lite_distill](mae_lite_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1-O7uuEnRrgKobETv54Z5T4isdS_WXOmK/view?usp=sharing) |
|  |  | [impr.](../eval_tools/finetuning_exp.py) | 300 | 82.5 | [link](https://drive.google.com/file/d/1ICgWu0V19TkDvpUutce3k8ahUAbJnEZP/view?usp=sharing) |
